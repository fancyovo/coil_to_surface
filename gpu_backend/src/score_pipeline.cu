#include "coil_field.h"
#include "coil_field_internal.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr double PI = 3.1415926535897932384626433832795;
constexpr double TWOPI = 2.0 * PI;
constexpr int SCORE_STAGE_NONE = 0;
constexpr int SCORE_STAGE_FIELD = 1;
constexpr int SCORE_STAGE_AXIS = 2;
constexpr int SCORE_STAGE_PSI = 3;
constexpr int SCORE_STAGE_SURFACE = 4;
constexpr int SCORE_STAGE_FLUX = 5;
constexpr int SCORE_STAGE_ALPHA = 6;
constexpr int SCORE_STAGE_QS = 7;
constexpr int SCORE_STAGE_COMPLETE = 8;

double seconds_since(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

double finite_or(double value, double fallback) {
    return std::isfinite(value) ? value : fallback;
}

double clip01(double value) {
    return std::min(1.0, std::max(0.0, finite_or(value, 0.0)));
}

double q_down(double value, double scale, double power, double fallback = 0.0) {
    if (!std::isfinite(value) || !(scale > 0.0)) return fallback;
    const double x = std::max(value, 0.0) / scale;
    return 1.0 / (1.0 + std::pow(x, power));
}

double q_up(double value, double scale, double power, double fallback = 0.0) {
    if (!std::isfinite(value) || value <= 0.0 || !(scale > 0.0)) return fallback;
    return 1.0 / (1.0 + std::pow(scale / value, power));
}

double blend(std::initializer_list<std::pair<double, double>> values) {
    double total = 0.0;
    double weighted = 0.0;
    for (const auto& item : values) {
        if (item.first > 0.0) {
            total += item.first;
            weighted += item.first * clip01(item.second);
        }
    }
    return total > 0.0 ? weighted / total : 0.0;
}

void initialize_result(SgpuScoreResult* result, int device_id) {
    std::memset(result, 0, sizeof(*result));
    result->abi_version = SGPU_SCORE_ABI_VERSION;
    result->struct_size = sizeof(*result);
    result->status = SGPU_SCORE_INTERNAL_ERROR;
    result->stage_completed = SCORE_STAGE_NONE;
    result->device_id = device_id;
    result->score = std::numeric_limits<double>::quiet_NaN();
    double* first_nan = &result->axis_R;
    double* last_nan = &result->coil_current_abs_max_a;
    for (double* ptr = first_nan; ptr <= last_nan; ++ptr) {
        *ptr = std::numeric_limits<double>::quiet_NaN();
    }
}

int fail_result(SgpuScoreResult* result, const char* message) {
    result->status = SGPU_SCORE_INTERNAL_ERROR;
    std::snprintf(result->error_message, sizeof(result->error_message), "%s", message ? message : "unknown error");
    sgpu_internal_set_error(result->error_message);
    return 1;
}

int fail_from_backend(SgpuScoreResult* result, const char* stage) {
    const char* backend = sgpu_last_error();
    std::string message = stage ? stage : "GPU backend failure";
    if (backend && backend[0]) {
        message += ": ";
        message += backend;
    }
    return fail_result(result, message.c_str());
}

double percentile(std::vector<double> values, double fraction) {
    values.erase(
        std::remove_if(values.begin(), values.end(), [](double x) { return !std::isfinite(x); }),
        values.end()
    );
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    const double position = clip01(fraction) * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(position));
    const size_t hi = static_cast<size_t>(std::ceil(position));
    const double alpha = position - static_cast<double>(lo);
    return values[lo] * (1.0 - alpha) + values[hi] * alpha;
}

void eval_fourier_interleaved(
    const double* coeff,
    int n_coeff,
    double t,
    double& value,
    double& derivative,
    double& second_derivative
) {
    const int order = (n_coeff - 1) / 2;
    value = coeff[0];
    derivative = 0.0;
    second_derivative = 0.0;
    for (int mode = 1; mode <= order; ++mode) {
        const double omega = TWOPI * static_cast<double>(mode);
        const double argument = omega * t;
        const double sine = std::sin(argument);
        const double cosine = std::cos(argument);
        const double sine_coeff = coeff[2 * mode - 1];
        const double cosine_coeff = coeff[2 * mode];
        value += sine_coeff * sine + cosine_coeff * cosine;
        derivative += omega * (sine_coeff * cosine - cosine_coeff * sine);
        second_derivative -= omega * omega * (sine_coeff * sine + cosine_coeff * cosine);
    }
}

struct Vec3d {
    double x;
    double y;
    double z;
};

Vec3d rotate_z(Vec3d point, double angle) {
    const double cosine = std::cos(angle);
    const double sine = std::sin(angle);
    return {
        cosine * point.x - sine * point.y,
        sine * point.x + cosine * point.y,
        point.z,
    };
}

struct CoilMetrics {
    double length_mean = 0.0;
    double curvature_p95 = 0.0;
    double curvature_max = 0.0;
    double min_intercoil_distance = std::numeric_limits<double>::infinity();
    double min_axis_distance = std::numeric_limits<double>::infinity();
    double high_mode_fraction = 0.0;
    double current_abs_max = 0.0;
};

CoilMetrics compute_coil_metrics(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp
) {
    constexpr int samples = 160;
    const int order = (n_coeff - 1) / 2;
    std::vector<std::vector<Vec3d>> base_points(n_base_coils);
    std::vector<double> lengths(n_base_coils, 0.0);
    std::vector<double> curvatures;
    curvatures.reserve(static_cast<size_t>(n_base_coils) * samples);
    for (int coil = 0; coil < n_base_coils; ++coil) {
        auto& points = base_points[coil];
        points.reserve(samples);
        for (int sample = 0; sample < samples; ++sample) {
            const double t = static_cast<double>(sample) / samples;
            double x, dx, ddx, y, dy, ddy, z, dz, ddz;
            eval_fourier_interleaved(coeffs_x + static_cast<size_t>(coil) * n_coeff, n_coeff, t, x, dx, ddx);
            eval_fourier_interleaved(coeffs_y + static_cast<size_t>(coil) * n_coeff, n_coeff, t, y, dy, ddy);
            eval_fourier_interleaved(coeffs_z + static_cast<size_t>(coil) * n_coeff, n_coeff, t, z, dz, ddz);
            points.push_back({x, y, z});
            const double speed2 = dx * dx + dy * dy + dz * dz;
            const double speed = std::sqrt(speed2);
            lengths[coil] += speed / samples;
            const double cx = dy * ddz - dz * ddy;
            const double cy = dz * ddx - dx * ddz;
            const double cz = dx * ddy - dy * ddx;
            curvatures.push_back(std::sqrt(cx * cx + cy * cy + cz * cz) /
                                 std::max(speed2 * speed, 1.0e-30));
        }
    }

    struct LabelledPoint {
        Vec3d point;
        int label;
    };
    std::vector<LabelledPoint> full_points;
    full_points.reserve(static_cast<size_t>(n_base_coils) * 2 * nfp * samples);
    int label = 0;
    for (const auto& base : base_points) {
        for (int reflected = 0; reflected < 2; ++reflected) {
            for (int period = 0; period < nfp; ++period) {
                const double angle = TWOPI * static_cast<double>(period) / nfp;
                for (Vec3d point : base) {
                    if (reflected) {
                        point.y = -point.y;
                        point.z = -point.z;
                    }
                    full_points.push_back({rotate_z(point, angle), label});
                }
                ++label;
            }
        }
    }

    CoilMetrics metrics;
    metrics.length_mean = std::accumulate(lengths.begin(), lengths.end(), 0.0) /
                          std::max(static_cast<int>(lengths.size()), 1);
    metrics.curvature_p95 = percentile(curvatures, 0.95);
    metrics.curvature_max = *std::max_element(curvatures.begin(), curvatures.end());
    for (size_t i = 0; i < full_points.size(); ++i) {
        const Vec3d& p = full_points[i].point;
        metrics.min_axis_distance = std::min(metrics.min_axis_distance, std::hypot(p.x, p.y));
        for (size_t j = i + 1; j < full_points.size(); ++j) {
            if (full_points[i].label == full_points[j].label) continue;
            const Vec3d& q = full_points[j].point;
            const double dx = p.x - q.x;
            const double dy = p.y - q.y;
            const double dz = p.z - q.z;
            metrics.min_intercoil_distance = std::min(
                metrics.min_intercoil_distance,
                std::sqrt(dx * dx + dy * dy + dz * dz)
            );
        }
    }
    double total_mode_energy = 0.0;
    double high_mode_energy = 0.0;
    const int high_start = std::max(1, static_cast<int>(std::floor(0.6 * order)));
    for (const double* block : {coeffs_x, coeffs_y, coeffs_z}) {
        for (int coil = 0; coil < n_base_coils; ++coil) {
            const double* coeff = block + static_cast<size_t>(coil) * n_coeff;
            for (int mode = 1; mode <= order; ++mode) {
                const double energy = coeff[2 * mode - 1] * coeff[2 * mode - 1] +
                                      coeff[2 * mode] * coeff[2 * mode];
                total_mode_energy += energy;
                if (mode >= high_start) high_mode_energy += energy;
            }
        }
    }
    metrics.high_mode_fraction = total_mode_energy > 0.0 ? high_mode_energy / total_mode_energy : 0.0;
    for (int coil = 0; coil < n_base_coils; ++coil) {
        metrics.current_abs_max = std::max(metrics.current_abs_max, std::abs(currents_a[coil]));
    }
    return metrics;
}

double coil_component(const CoilMetrics& metrics) {
    const double length = q_down(metrics.length_mean, 7.0, 1.4, 0.6);
    const double curvature_p95 = q_down(metrics.curvature_p95, 10.0, 1.3, 0.5);
    const double curvature_max = q_down(metrics.curvature_max, 35.0, 1.2, 0.5);
    const double spacing = q_up(metrics.min_intercoil_distance, 0.08, 1.1, 0.45);
    const double axis_distance = q_up(metrics.min_axis_distance, 0.20, 1.2, 0.45);
    const double high_mode = q_down(metrics.high_mode_fraction, 0.05, 1.0, 0.7);
    const double current = q_down(metrics.current_abs_max, 2.0e6, 1.0, 0.7);
    return blend({
        {0.16, length}, {0.20, curvature_p95}, {0.12, curvature_max},
        {0.20, spacing}, {0.12, axis_distance}, {0.13, high_mode}, {0.07, current},
    });
}

struct AxisDomain {
    double r_min;
    double r_max;
    double z_min;
    double z_max;
};

AxisDomain build_axis_domain(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    int n_base_coils,
    int n_coeff,
    const SgpuScoreConfig& config
) {
    std::vector<double> radii;
    std::vector<double> abs_z;
    radii.reserve(static_cast<size_t>(n_base_coils) * 160);
    abs_z.reserve(radii.capacity());
    double center = 0.0;
    for (int coil = 0; coil < n_base_coils; ++coil) {
        center += std::hypot(coeffs_x[static_cast<size_t>(coil) * n_coeff],
                             coeffs_y[static_cast<size_t>(coil) * n_coeff]);
        for (int sample = 0; sample < 160; ++sample) {
            const double t = static_cast<double>(sample) / 160.0;
            double x, dx, ddx, y, dy, ddy, z, dz, ddz;
            eval_fourier_interleaved(coeffs_x + static_cast<size_t>(coil) * n_coeff, n_coeff, t, x, dx, ddx);
            eval_fourier_interleaved(coeffs_y + static_cast<size_t>(coil) * n_coeff, n_coeff, t, y, dy, ddy);
            eval_fourier_interleaved(coeffs_z + static_cast<size_t>(coil) * n_coeff, n_coeff, t, z, dz, ddz);
            radii.push_back(std::hypot(x, y));
            abs_z.push_back(std::abs(z));
        }
    }
    center /= std::max(n_base_coils, 1);
    const double r05 = percentile(radii, 0.05);
    const double r95 = percentile(radii, 0.95);
    const double z95 = percentile(abs_z, 0.95);
    const double r_span = std::max(config.axis_span, 0.45 * (r95 - r05));
    const double z_span = std::max(config.axis_span, 1.15 * z95);
    AxisDomain domain;
    domain.r_min = std::max(config.axis_r_floor, std::min(center - config.axis_span, center - r_span));
    domain.r_max = std::max(center + config.axis_span, center + r_span);
    domain.z_min = -z_span;
    domain.z_max = z_span;
    if (!(domain.r_max > domain.r_min)) domain.r_max = domain.r_min + std::max(config.axis_span, 0.1);
    return domain;
}

bool trace_map(
    void* field,
    const std::vector<double>& R,
    const std::vector<double>& Z,
    int nfp,
    int steps,
    bool verify_fp64,
    std::vector<double>& R_end,
    std::vector<double>& Z_end
) {
    if (R.size() != Z.size() || R.empty()) return false;
    R_end.resize(R.size());
    Z_end.resize(Z.size());
    const int code = verify_fp64
        ? sgpu_trace_period_blockline(
              field, R.data(), Z.data(), R_end.data(), Z_end.data(),
              static_cast<int>(R.size()), nfp, steps, 256)
        : sgpu_trace_period_blockline_mixed(
              field, R.data(), Z.data(), R_end.data(), Z_end.data(),
              static_cast<int>(R.size()), nfp, steps, 256, 1);
    return code == 0;
}

struct AxisCandidate {
    double R = 0.0;
    double Z = 0.0;
    double residual = std::numeric_limits<double>::infinity();
    bool winding = false;
    double topology_trace = std::numeric_limits<double>::quiet_NaN();
    double topology_det = std::numeric_limits<double>::quiet_NaN();
    double ellipse_aspect = std::numeric_limits<double>::infinity();
    bool elliptic = false;
};

double wrapped_angle_delta(double from, double to) {
    double delta = to - from;
    while (delta <= -PI) delta += TWOPI;
    while (delta > PI) delta -= TWOPI;
    return delta;
}

std::vector<AxisCandidate> find_grid_candidates(
    const std::vector<double>& rs,
    const std::vector<double>& zs,
    const std::vector<double>& dR,
    const std::vector<double>& dZ,
    int max_candidates
) {
    const int grid = static_cast<int>(rs.size());
    std::vector<double> angle(static_cast<size_t>(grid) * grid);
    std::vector<double> residual(angle.size());
    for (size_t index = 0; index < angle.size(); ++index) {
        angle[index] = std::atan2(dZ[index], dR[index]);
        residual[index] = std::hypot(dR[index], dZ[index]);
    }
    std::vector<AxisCandidate> candidates;
    for (int j = 0; j + 1 < grid; ++j) {
        for (int i = 0; i + 1 < grid; ++i) {
            const size_t a = static_cast<size_t>(j) * grid + i;
            const size_t b = a + 1;
            const size_t d = static_cast<size_t>(j + 1) * grid + i;
            const size_t c = d + 1;
            const double winding = wrapped_angle_delta(angle[a], angle[b]) +
                                   wrapped_angle_delta(angle[b], angle[c]) +
                                   wrapped_angle_delta(angle[c], angle[d]) +
                                   wrapped_angle_delta(angle[d], angle[a]);
            if (std::abs(winding) > PI) {
                candidates.push_back({
                    0.5 * (rs[i] + rs[i + 1]),
                    0.5 * (zs[j] + zs[j + 1]),
                    std::min({residual[a], residual[b], residual[c], residual[d]}),
                    true,
                });
            }
        }
    }
    std::vector<AxisCandidate> local_minima;
    for (int j = 1; j + 1 < grid; ++j) {
        for (int i = 1; i + 1 < grid; ++i) {
            const size_t center = static_cast<size_t>(j) * grid + i;
            bool minimum = std::isfinite(residual[center]);
            for (int dj = -1; dj <= 1 && minimum; ++dj) {
                for (int di = -1; di <= 1; ++di) {
                    if (residual[center] > residual[static_cast<size_t>(j + dj) * grid + i + di]) {
                        minimum = false;
                        break;
                    }
                }
            }
            if (minimum) local_minima.push_back({rs[i], zs[j], residual[center], false});
        }
    }
    std::sort(local_minima.begin(), local_minima.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.residual < rhs.residual;
    });
    const size_t local_limit = std::min<size_t>(64, local_minima.size());
    candidates.insert(candidates.end(), local_minima.begin(), local_minima.begin() + local_limit);
    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.winding != rhs.winding) return lhs.winding > rhs.winding;
        return lhs.residual < rhs.residual;
    });
    const double step = std::max(
        grid > 1 ? rs[1] - rs[0] : 0.0,
        grid > 1 ? zs[1] - zs[0] : 0.0
    );
    std::vector<AxisCandidate> unique;
    for (const auto& candidate : candidates) {
        bool separated = true;
        for (const auto& previous : unique) {
            if (std::hypot(candidate.R - previous.R, candidate.Z - previous.Z) < 0.75 * step) {
                separated = false;
                break;
            }
        }
        if (separated) unique.push_back(candidate);
        if (static_cast<int>(unique.size()) >= max_candidates) break;
    }
    return unique;
}

bool refine_axis_candidates(
    void* field,
    std::vector<AxisCandidate>& candidates,
    const AxisDomain& domain,
    int nfp,
    const SgpuScoreConfig& config,
    int iterations
) {
    if (candidates.empty()) return true;
    std::vector<double> R(candidates.size()), Z(candidates.size()), R_end, Z_end;
    for (size_t i = 0; i < candidates.size(); ++i) {
        R[i] = candidates[i].R;
        Z[i] = candidates[i].Z;
    }
    if (!trace_map(field, R, Z, nfp, config.axis_trace_steps, false, R_end, Z_end)) return false;
    std::vector<double> fR(R.size()), fZ(Z.size()), residual(R.size());
    for (size_t i = 0; i < R.size(); ++i) {
        fR[i] = R_end[i] - R[i];
        fZ[i] = Z_end[i] - Z[i];
        residual[i] = std::hypot(fR[i], fZ[i]);
    }
    const double span = std::max(domain.r_max - domain.r_min, domain.z_max - domain.z_min);
    const double h = std::max(config.axis_fd_absolute, config.axis_fd_relative * span);
    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::vector<size_t> active;
        for (size_t i = 0; i < R.size(); ++i) {
            if (residual[i] > config.axis_tolerance) active.push_back(i);
        }
        if (active.empty()) break;
        const size_t count = active.size();
        std::vector<double> eval_R(4 * count), eval_Z(4 * count), eval_R_end, eval_Z_end;
        for (size_t j = 0; j < count; ++j) {
            const size_t i = active[j];
            eval_R[j] = std::min(domain.r_max, R[i] + h);
            eval_Z[j] = Z[i];
            eval_R[count + j] = std::max(domain.r_min, R[i] - h);
            eval_Z[count + j] = Z[i];
            eval_R[2 * count + j] = R[i];
            eval_Z[2 * count + j] = std::min(domain.z_max, Z[i] + h);
            eval_R[3 * count + j] = R[i];
            eval_Z[3 * count + j] = std::max(domain.z_min, Z[i] - h);
        }
        if (!trace_map(field, eval_R, eval_Z, nfp, config.axis_trace_steps, false, eval_R_end, eval_Z_end)) return false;
        std::vector<double> step_R(count, 0.0), step_Z(count, 0.0);
        for (size_t j = 0; j < count; ++j) {
            const size_t i = active[j];
            auto map_residual_R = [&](size_t k) { return eval_R_end[k] - eval_R[k]; };
            auto map_residual_Z = [&](size_t k) { return eval_Z_end[k] - eval_Z[k]; };
            const double denom_R = std::max(eval_R[j] - eval_R[count + j], 1.0e-300);
            const double denom_Z = std::max(eval_Z[2 * count + j] - eval_Z[3 * count + j], 1.0e-300);
            const double j11 = (map_residual_R(j) - map_residual_R(count + j)) / denom_R;
            const double j21 = (map_residual_Z(j) - map_residual_Z(count + j)) / denom_R;
            const double j12 = (map_residual_R(2 * count + j) - map_residual_R(3 * count + j)) / denom_Z;
            const double j22 = (map_residual_Z(2 * count + j) - map_residual_Z(3 * count + j)) / denom_Z;
            const double determinant = j11 * j22 - j12 * j21;
            if (std::abs(determinant) > 1.0e-14) {
                step_R[j] = (-fR[i] * j22 + j12 * fZ[i]) / determinant;
                step_Z[j] = (j21 * fR[i] - j11 * fZ[i]) / determinant;
                const double norm = std::hypot(step_R[j], step_Z[j]);
                const double scale = std::min(1.0, 0.25 * span / std::max(norm, 1.0e-300));
                step_R[j] *= scale;
                step_Z[j] *= scale;
            }
        }
        std::vector<unsigned char> accepted(count, 0);
        for (double alpha : {1.0, 0.5, 0.25, 0.125}) {
            std::vector<double> trial_R(count), trial_Z(count), trial_R_end, trial_Z_end;
            for (size_t j = 0; j < count; ++j) {
                const size_t i = active[j];
                trial_R[j] = std::min(domain.r_max, std::max(domain.r_min, R[i] + alpha * step_R[j]));
                trial_Z[j] = std::min(domain.z_max, std::max(domain.z_min, Z[i] + alpha * step_Z[j]));
            }
            if (!trace_map(field, trial_R, trial_Z, nfp, config.axis_trace_steps, false, trial_R_end, trial_Z_end)) return false;
            for (size_t j = 0; j < count; ++j) {
                if (accepted[j]) continue;
                const size_t i = active[j];
                const double trial_fR = trial_R_end[j] - trial_R[j];
                const double trial_fZ = trial_Z_end[j] - trial_Z[j];
                const double trial_residual = std::hypot(trial_fR, trial_fZ);
                if (std::isfinite(trial_residual) && trial_residual < residual[i]) {
                    R[i] = trial_R[j];
                    Z[i] = trial_Z[j];
                    fR[i] = trial_fR;
                    fZ[i] = trial_fZ;
                    residual[i] = trial_residual;
                    accepted[j] = 1;
                }
            }
        }
    }
    for (size_t i = 0; i < candidates.size(); ++i) {
        candidates[i].R = R[i];
        candidates[i].Z = Z[i];
        candidates[i].residual = residual[i];
    }
    return true;
}

void classify_axis_topology(
    void* field,
    std::vector<AxisCandidate>& candidates,
    const AxisDomain& domain,
    int nfp,
    const SgpuScoreConfig& config
) {
    if (candidates.empty()) return;
    const size_t count = std::min<size_t>(8, candidates.size());
    const double span = std::max(domain.r_max - domain.r_min, domain.z_max - domain.z_min);
    const double h = std::max(config.axis_fd_absolute, config.axis_fd_relative * span);
    std::vector<double> R(4 * count), Z(4 * count), R_end, Z_end;
    for (size_t j = 0; j < count; ++j) {
        R[j] = std::min(domain.r_max, candidates[j].R + h);
        Z[j] = candidates[j].Z;
        R[count + j] = std::max(domain.r_min, candidates[j].R - h);
        Z[count + j] = candidates[j].Z;
        R[2 * count + j] = candidates[j].R;
        Z[2 * count + j] = std::min(domain.z_max, candidates[j].Z + h);
        R[3 * count + j] = candidates[j].R;
        Z[3 * count + j] = std::max(domain.z_min, candidates[j].Z - h);
    }
    if (!trace_map(field, R, Z, nfp, config.axis_trace_steps, true, R_end, Z_end)) return;
    for (size_t j = 0; j < count; ++j) {
        const double denom_R = std::max(R[j] - R[count + j], 1.0e-300);
        const double denom_Z = std::max(Z[2 * count + j] - Z[3 * count + j], 1.0e-300);
        const double a = (R_end[j] - R_end[count + j]) / denom_R;
        const double c = (Z_end[j] - Z_end[count + j]) / denom_R;
        const double b = (R_end[2 * count + j] - R_end[3 * count + j]) / denom_Z;
        const double d = (Z_end[2 * count + j] - Z_end[3 * count + j]) / denom_Z;
        const double trace = a + d;
        const double determinant = a * d - b * c;
        candidates[j].topology_trace = trace;
        candidates[j].topology_det = determinant;
        candidates[j].elliptic = determinant > 0.0 &&
            std::abs(trace / std::sqrt(determinant)) < 2.0 - config.axis_topology_margin;
        if (determinant > 0.0) {
            const double scale = std::sqrt(determinant);
            const double q00 = c / scale;
            const double q01 = 0.5 * (d - a) / scale;
            const double q11 = -b / scale;
            const double middle = 0.5 * (q00 + q11);
            const double radius = std::hypot(0.5 * (q00 - q11), q01);
            double eig0 = middle - radius;
            double eig1 = middle + radius;
            if (eig0 < 0.0 && eig1 < 0.0) {
                const double old0 = eig0;
                eig0 = -eig1;
                eig1 = -old0;
            }
            if (eig0 > 0.0 && eig1 >= eig0) {
                candidates[j].ellipse_aspect = std::sqrt(eig1 / eig0);
            }
        }
    }
}

bool search_axis_grid(
    void* field,
    const AxisDomain& domain,
    int nfp,
    const SgpuScoreConfig& config,
    int grid,
    int max_candidates,
    int newton_iters,
    std::vector<AxisCandidate>& candidates
) {
    std::vector<double> rs(grid), zs(grid), R(static_cast<size_t>(grid) * grid), Z(R.size());
    for (int i = 0; i < grid; ++i) {
        rs[i] = domain.r_min + (domain.r_max - domain.r_min) * i / std::max(grid - 1, 1);
        zs[i] = domain.z_min + (domain.z_max - domain.z_min) * i / std::max(grid - 1, 1);
    }
    for (int j = 0; j < grid; ++j) {
        for (int i = 0; i < grid; ++i) {
            const size_t index = static_cast<size_t>(j) * grid + i;
            R[index] = rs[i];
            Z[index] = zs[j];
        }
    }
    std::vector<double> R_end, Z_end;
    if (!trace_map(field, R, Z, nfp, config.axis_trace_steps, false, R_end, Z_end)) return false;
    std::vector<double> dR(R.size()), dZ(Z.size());
    for (size_t i = 0; i < R.size(); ++i) {
        dR[i] = R_end[i] - R[i];
        dZ[i] = Z_end[i] - Z[i];
    }
    candidates = find_grid_candidates(rs, zs, dR, dZ, max_candidates);
    if (!refine_axis_candidates(field, candidates, domain, nfp, config, newton_iters)) return false;
    std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.residual < rhs.residual;
    });
    if (candidates.size() > 8) candidates.resize(8);
    if (!candidates.empty()) {
        std::vector<double> verify_R(candidates.size()), verify_Z(candidates.size()), end_R, end_Z;
        for (size_t i = 0; i < candidates.size(); ++i) {
            verify_R[i] = candidates[i].R;
            verify_Z[i] = candidates[i].Z;
        }
        if (!trace_map(field, verify_R, verify_Z, nfp, config.axis_trace_steps, true, end_R, end_Z)) return false;
        for (size_t i = 0; i < candidates.size(); ++i) {
            candidates[i].residual = std::hypot(end_R[i] - verify_R[i], end_Z[i] - verify_Z[i]);
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.residual < rhs.residual;
        });
        classify_axis_topology(field, candidates, domain, nfp, config);
    }
    return true;
}

struct AxisData {
    AxisCandidate selected;
    std::vector<double> R;
    std::vector<double> Z;
    std::vector<double> R_phi;
    std::vector<double> Z_phi;
    int candidate_count = 0;
};

bool find_axis_native(
    void* field,
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig& config,
    AxisData& axis
) {
    const AxisDomain domain = build_axis_domain(
        coeffs_x, coeffs_y, coeffs_z, n_base_coils, n_coeff, config
    );
    std::vector<AxisCandidate> candidates;
    if (!search_axis_grid(
            field, domain, nfp, config, config.axis_grid,
            config.axis_max_candidates, config.axis_newton_iters, candidates)) {
        return false;
    }
    auto eligible = [](const AxisCandidate& candidate, double tolerance) {
        return candidate.residual <= tolerance && candidate.elliptic;
    };
    bool has_eligible = std::any_of(candidates.begin(), candidates.end(), [&](const auto& candidate) {
        return eligible(candidate, config.axis_tolerance);
    });
    if (!has_eligible) {
        std::vector<AxisCandidate> fallback;
        if (!search_axis_grid(
                field, domain, nfp, config, config.axis_fallback_grid,
                config.axis_fallback_max_candidates, config.axis_fallback_newton_iters,
                fallback)) {
            return false;
        }
        axis.candidate_count = static_cast<int>(candidates.size() + fallback.size());
        candidates.insert(candidates.end(), fallback.begin(), fallback.end());
    } else {
        axis.candidate_count = static_cast<int>(candidates.size());
    }
    std::vector<AxisCandidate> elliptic;
    for (const auto& candidate : candidates) {
        if (eligible(candidate, config.axis_tolerance)) elliptic.push_back(candidate);
    }
    if (elliptic.empty()) return true;
    axis.selected = *std::min_element(elliptic.begin(), elliptic.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.ellipse_aspect != rhs.ellipse_aspect) return lhs.ellipse_aspect < rhs.ellipse_aspect;
        return lhs.residual < rhs.residual;
    });
    axis.R.resize(config.axis_sample_count);
    axis.Z.resize(config.axis_sample_count);
    axis.R_phi.resize(config.axis_sample_count);
    axis.Z_phi.resize(config.axis_sample_count);
    if (sgpu_trace_axis_samples(
            field, axis.selected.R, axis.selected.Z, nfp,
            config.axis_trace_steps, config.axis_sample_count,
            axis.R.data(), axis.Z.data(), axis.R_phi.data(), axis.Z_phi.data())) {
        return false;
    }
    return true;
}

void periodic_hermite_host(
    double phi,
    const std::vector<double>& values,
    const std::vector<double>& derivatives,
    int nfp,
    double& value,
    double& derivative
) {
    const int count = static_cast<int>(values.size());
    const double period = TWOPI / nfp;
    double wrapped = std::fmod(phi, period);
    if (wrapped < 0.0) wrapped += period;
    const double position = wrapped * count / period;
    const int index0 = std::min(static_cast<int>(std::floor(position)), count - 1);
    const int index1 = (index0 + 1) % count;
    const double t = position - index0;
    const double h = period / count;
    const double h00 = (2.0 * t - 3.0) * t * t + 1.0;
    const double h10 = ((t - 2.0) * t + 1.0) * t;
    const double h01 = (-2.0 * t + 3.0) * t * t;
    const double h11 = (t - 1.0) * t * t;
    value = h00 * values[index0] + h10 * h * derivatives[index0] +
            h01 * values[index1] + h11 * h * derivatives[index1];
    const double dh00 = (6.0 * t - 6.0) * t;
    const double dh10 = (3.0 * t - 4.0) * t + 1.0;
    const double dh01 = (-6.0 * t + 6.0) * t;
    const double dh11 = (3.0 * t - 2.0) * t;
    derivative = (dh00 * values[index0] + dh01 * values[index1]) / h +
                 dh10 * derivatives[index0] + dh11 * derivatives[index1];
}

struct PsiModes {
    std::vector<int> a;
    std::vector<int> b;
    std::vector<int> m;
    std::vector<int> kind;
};

PsiModes build_psi_modes(int degree, int m_tor) {
    PsiModes modes;
    for (int total = 2; total <= degree; ++total) {
        for (int a = total; a >= 0; --a) {
            const int b = total - a;
            for (int m = 0; m <= m_tor; ++m) {
                if (a == 2 && b == 0 && m == 0) continue;
                modes.a.push_back(a);
                modes.b.push_back(b);
                modes.m.push_back(m);
                modes.kind.push_back(0);
                if (m > 0) {
                    modes.a.push_back(a);
                    modes.b.push_back(b);
                    modes.m.push_back(m);
                    modes.kind.push_back(1);
                }
            }
        }
    }
    return modes;
}

struct PsiData {
    PsiModes modes;
    std::vector<double> coeffs;
    double train_rms = std::numeric_limits<double>::quiet_NaN();
    double angle_mean = std::numeric_limits<double>::quiet_NaN();
    double angle_p95 = std::numeric_limits<double>::quiet_NaN();
    double angle_l2 = std::numeric_limits<double>::quiet_NaN();
};

void evaluate_psi_host(
    const PsiData& psi,
    const AxisData& axis,
    const SgpuScoreConfig& config,
    int nfp,
    double R,
    double Z,
    double phi,
    double& value,
    double& grad_R,
    double& grad_Z,
    double& grad_phi
) {
    double axis_R, axis_R_phi, axis_Z, axis_Z_phi;
    periodic_hermite_host(phi, axis.R, axis.R_phi, nfp, axis_R, axis_R_phi);
    periodic_hermite_host(phi, axis.Z, axis.Z_phi, nfp, axis_Z, axis_Z_phi);
    const double X = (R - axis_R) / config.psi_a;
    const double Y = (Z - axis_Z) / config.psi_a;
    std::array<double, 25> xpow{}, ypow{};
    xpow[0] = 1.0;
    ypow[0] = 1.0;
    for (int power = 1; power <= config.psi_poly_degree; ++power) {
        xpow[power] = xpow[power - 1] * X;
        ypow[power] = ypow[power - 1] * Y;
    }
    value = X * X;
    grad_R = 2.0 * X / config.psi_a;
    grad_Z = 0.0;
    grad_phi = -axis_R_phi * grad_R;
    for (size_t k = 0; k < psi.coeffs.size(); ++k) {
        const int a = psi.modes.a[k];
        const int b = psi.modes.b[k];
        const int m = psi.modes.m[k];
        const double argument = static_cast<double>(m * nfp) * phi;
        const double trig = m == 0 ? 1.0 : (psi.modes.kind[k] == 0 ? std::cos(argument) : std::sin(argument));
        const double trig_phi = m == 0 ? 0.0 : static_cast<double>(m * nfp) *
            (psi.modes.kind[k] == 0 ? -std::sin(argument) : std::cos(argument));
        const double mono = xpow[a] * ypow[b];
        const double derivative_x = a > 0 ? a * xpow[a - 1] * ypow[b] / config.psi_a : 0.0;
        const double derivative_y = b > 0 ? b * xpow[a] * ypow[b - 1] / config.psi_a : 0.0;
        const double coefficient = psi.coeffs[k];
        value += coefficient * mono * trig;
        grad_R += coefficient * derivative_x * trig;
        grad_Z += coefficient * derivative_y * trig;
        grad_phi += coefficient * (mono * trig_phi - axis_R_phi * derivative_x * trig - axis_Z_phi * derivative_y * trig);
    }
}

bool fit_psi_native(
    void* field,
    const AxisData& axis,
    int nfp,
    const SgpuScoreConfig& config,
    PsiData& psi,
    double& point_generation_time,
    double& fit_time
) {
    auto started = Clock::now();
    std::vector<double> R, Z, phi;
    const size_t reserve = static_cast<size_t>(config.psi_n_r) * config.psi_n_z * config.psi_n_phi;
    R.reserve(reserve);
    Z.reserve(reserve);
    phi.reserve(reserve);
    for (int ir = 0; ir < config.psi_n_r; ++ir) {
        const double dR = -config.psi_a + 2.0 * config.psi_a * ir / std::max(config.psi_n_r - 1, 1);
        for (int iz = 0; iz < config.psi_n_z; ++iz) {
            const double dZ = -config.psi_a + 2.0 * config.psi_a * iz / std::max(config.psi_n_z - 1, 1);
            const double radius = std::hypot(dR, dZ);
            if (radius < config.psi_rho_min || radius > config.psi_a) continue;
            for (int iphi = 0; iphi < config.psi_n_phi; ++iphi) {
                const double angle = TWOPI * iphi / static_cast<double>(nfp * config.psi_n_phi);
                double axis_R, axis_R_phi, axis_Z, axis_Z_phi;
                periodic_hermite_host(angle, axis.R, axis.R_phi, nfp, axis_R, axis_R_phi);
                periodic_hermite_host(angle, axis.Z, axis.Z_phi, nfp, axis_Z, axis_Z_phi);
                R.push_back(axis_R + dR);
                Z.push_back(axis_Z + dZ);
                phi.push_back(angle);
            }
        }
    }
    point_generation_time = seconds_since(started);
    psi.modes = build_psi_modes(config.psi_poly_degree, config.psi_m_tor);
    psi.coeffs.resize(psi.modes.a.size());
    std::array<double, 16> stats{};
    started = Clock::now();
    const int code = sgpu_fit_psi_fullgpu(
        field, R.data(), Z.data(), phi.data(), static_cast<int>(R.size()),
        axis.R.data(), axis.Z.data(), axis.R_phi.data(), axis.Z_phi.data(),
        static_cast<int>(axis.R.size()),
        psi.modes.a.data(), psi.modes.b.data(), psi.modes.m.data(), psi.modes.kind.data(),
        static_cast<int>(psi.modes.a.size()), nfp, config.psi_a,
        config.psi_poly_degree, config.psi_m_tor, config.psi_ridge,
        2, 2, psi.coeffs.data(), &psi.train_rms, stats.data(), static_cast<int>(stats.size())
    );
    fit_time = seconds_since(started);
    return code == 0;
}

bool validate_psi_native(
    void* field,
    const AxisData& axis,
    int nfp,
    const SgpuScoreConfig& config,
    PsiData& psi
) {
    const int count = config.psi_validation_points;
    std::vector<float> xyz(static_cast<size_t>(count) * 3), B(xyz.size());
    std::vector<double> R(count), Z(count), phi(count), grad_norm(count), residual(count), angle(count);
    std::uint64_t state = 20260704u;
    auto uniform = [&]() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<double>(state >> 11) * (1.0 / 9007199254740992.0);
    };
    const double u_min = std::pow(config.psi_rho_min / config.psi_a, 2.0);
    for (int index = 0; index < count; ++index) {
        phi[index] = uniform() * TWOPI / nfp;
        const double theta = uniform() * TWOPI;
        const double radius = config.psi_a * std::sqrt(u_min + (1.0 - u_min) * uniform());
        double axis_R, axis_R_phi, axis_Z, axis_Z_phi;
        periodic_hermite_host(phi[index], axis.R, axis.R_phi, nfp, axis_R, axis_R_phi);
        periodic_hermite_host(phi[index], axis.Z, axis.Z_phi, nfp, axis_Z, axis_Z_phi);
        R[index] = axis_R + radius * std::cos(theta);
        Z[index] = axis_Z + radius * std::sin(theta);
        xyz[3 * index] = static_cast<float>(R[index] * std::cos(phi[index]));
        xyz[3 * index + 1] = static_cast<float>(R[index] * std::sin(phi[index]));
        xyz[3 * index + 2] = static_cast<float>(Z[index]);
    }
    if (sgpu_eval_B_f32(field, xyz.data(), B.data(), count)) return false;
    double residual_norm2 = 0.0;
    double scale_norm2 = 0.0;
    double angle_sum = 0.0;
    for (int index = 0; index < count; ++index) {
        double value, gR, gZ, gPhi;
        evaluate_psi_host(psi, axis, config, nfp, R[index], Z[index], phi[index], value, gR, gZ, gPhi);
        const double cosine = std::cos(phi[index]);
        const double sine = std::sin(phi[index]);
        const double bx = B[3 * index];
        const double by = B[3 * index + 1];
        const double bz = B[3 * index + 2];
        const double br = bx * cosine + by * sine;
        const double bphi = -bx * sine + by * cosine;
        const double dot = br * gR + bz * gZ + bphi * gPhi / R[index];
        const double bnorm = std::sqrt(bx * bx + by * by + bz * bz);
        const double gnorm = std::sqrt(gR * gR + gZ * gZ + std::pow(gPhi / R[index], 2.0));
        residual[index] = dot;
        grad_norm[index] = bnorm * gnorm;
        angle[index] = std::abs(dot) / std::max(grad_norm[index], 1.0e-14);
        residual_norm2 += dot * dot;
        scale_norm2 += grad_norm[index] * grad_norm[index];
        angle_sum += angle[index];
    }
    psi.angle_mean = angle_sum / count;
    psi.angle_p95 = percentile(angle, 0.95);
    psi.angle_l2 = std::sqrt(residual_norm2 / std::max(scale_norm2, 1.0e-300));
    return true;
}

struct SurfaceScreen {
    double level = std::numeric_limits<double>::quiet_NaN();
    double drift_p95 = std::numeric_limits<double>::infinity();
    double relative_drift_p95 = std::numeric_limits<double>::infinity();
    double radius_mean = std::numeric_limits<double>::quiet_NaN();
    double radius_max = std::numeric_limits<double>::quiet_NaN();
    bool stable = false;
    bool strict = false;
};

void ray_polynomial_phi0(
    const PsiData& psi,
    const SgpuScoreConfig& config,
    double theta,
    std::array<double, 25>& polynomial
) {
    polynomial.fill(0.0);
    const double cosine = std::cos(theta);
    const double sine = std::sin(theta);
    std::array<double, 25> cpow{}, spow{};
    cpow[0] = 1.0;
    spow[0] = 1.0;
    for (int degree = 1; degree <= config.psi_poly_degree; ++degree) {
        cpow[degree] = cpow[degree - 1] * cosine;
        spow[degree] = spow[degree - 1] * sine;
    }
    polynomial[2] = cosine * cosine;
    for (size_t k = 0; k < psi.coeffs.size(); ++k) {
        if (psi.modes.kind[k] != 0) continue;
        polynomial[psi.modes.a[k] + psi.modes.b[k]] +=
            psi.coeffs[k] * cpow[psi.modes.a[k]] * spow[psi.modes.b[k]];
    }
}

double solve_ray_radius(
    const std::array<double, 25>& polynomial,
    double level,
    const SgpuScoreConfig& config
) {
    const double max_radius = config.surface_max_radius_scale * config.psi_a;
    const double q = polynomial[2];
    double radius = q > 1.0e-10
        ? config.psi_a * std::sqrt(std::max(level, 0.0) / q)
        : config.psi_a * std::sqrt(std::max(level, 1.0e-16));
    radius = std::min(max_radius, std::max(1.0e-12 * config.psi_a, radius));
    for (int iteration = 0; iteration < config.surface_newton_iters; ++iteration) {
        const double u = radius / config.psi_a;
        double value = 0.0;
        double derivative = 0.0;
        double power = u * u;
        double previous_power = u;
        for (int degree = 2; degree <= config.psi_poly_degree; ++degree) {
            if (degree > 2) {
                previous_power = power;
                power *= u;
            }
            value += polynomial[degree] * power;
            derivative += degree * polynomial[degree] * previous_power / config.psi_a;
        }
        const double residual = value - level;
        if (std::abs(residual) <= config.surface_newton_tolerance) break;
        const double denominator = std::abs(derivative) > 1.0e-14
            ? derivative : std::copysign(1.0e-14, derivative == 0.0 ? 1.0 : derivative);
        const double limit = 0.45 * std::max(std::abs(radius), 1.0e-8 * config.psi_a);
        const double step = std::min(limit, std::max(-limit, residual / denominator));
        radius = std::min(max_radius, std::max(1.0e-12 * config.psi_a, radius - step));
    }
    return radius;
}

bool screen_surfaces_native(
    void* field,
    const AxisData& axis,
    const PsiData& psi,
    int nfp,
    const SgpuScoreConfig& config,
    std::vector<SurfaceScreen>& screens
) {
    const int levels = config.surface_level_count;
    const int theta_count = config.surface_theta_count;
    const size_t count = static_cast<size_t>(levels) * theta_count;
    std::vector<double> R(count), Z(count), radii(count), theta(count), R_end, Z_end;
    for (int level_index = 0; level_index < levels; ++level_index) {
        for (int theta_index = 0; theta_index < theta_count; ++theta_index) {
            const size_t index = static_cast<size_t>(level_index) * theta_count + theta_index;
            theta[index] = TWOPI * theta_index / theta_count;
            std::array<double, 25> polynomial;
            ray_polynomial_phi0(psi, config, theta[index], polynomial);
            radii[index] = solve_ray_radius(polynomial, config.surface_levels[level_index], config);
            R[index] = axis.R[0] + radii[index] * std::cos(theta[index]);
            Z[index] = axis.Z[0] + radii[index] * std::sin(theta[index]);
        }
    }
    if (!trace_map(field, R, Z, nfp, config.surface_trace_steps, false, R_end, Z_end)) return false;
    screens.resize(levels);
    for (int level_index = 0; level_index < levels; ++level_index) {
        std::vector<double> distances;
        distances.reserve(theta_count);
        double radius_sum = 0.0;
        double radius_max = 0.0;
        for (int theta_index = 0; theta_index < theta_count; ++theta_index) {
            const size_t index = static_cast<size_t>(level_index) * theta_count + theta_index;
            double value, gR, gZ, gPhi;
            evaluate_psi_host(
                psi, axis, config, nfp, R_end[index], Z_end[index], TWOPI / nfp,
                value, gR, gZ, gPhi
            );
            const double gradient_norm = std::sqrt(gR * gR + gZ * gZ + std::pow(gPhi / R_end[index], 2.0));
            distances.push_back(std::abs(value - config.surface_levels[level_index]) /
                                std::max(gradient_norm, 1.0e-14));
            radius_sum += radii[index];
            radius_max = std::max(radius_max, radii[index]);
        }
        SurfaceScreen& screen = screens[level_index];
        screen.level = config.surface_levels[level_index];
        screen.drift_p95 = percentile(distances, 0.95);
        screen.radius_mean = radius_sum / theta_count;
        screen.radius_max = radius_max;
        screen.relative_drift_p95 = screen.drift_p95 / std::max(screen.radius_mean, 1.0e-14);
        screen.stable = screen.drift_p95 <= config.surface_drift_absolute_tolerance &&
                        screen.relative_drift_p95 <= 0.30 &&
                        screen.radius_max < config.surface_max_radius_scale * config.psi_a * 0.999;
        screen.strict = screen.stable &&
                        screen.relative_drift_p95 <= config.surface_drift_relative_tolerance;
    }

    std::vector<int> verify_indices;
    for (int index = levels - 1; index >= 0 && verify_indices.size() < 3; --index) {
        if (screens[index].strict) verify_indices.push_back(index);
    }
    if (!verify_indices.empty()) {
        std::vector<double> verify_R, verify_Z, verify_R_end, verify_Z_end;
        verify_R.reserve(static_cast<size_t>(verify_indices.size()) * theta_count);
        verify_Z.reserve(verify_R.capacity());
        for (int level_index : verify_indices) {
            const size_t offset = static_cast<size_t>(level_index) * theta_count;
            verify_R.insert(verify_R.end(), R.begin() + offset, R.begin() + offset + theta_count);
            verify_Z.insert(verify_Z.end(), Z.begin() + offset, Z.begin() + offset + theta_count);
        }
        if (!trace_map(field, verify_R, verify_Z, nfp, config.surface_trace_steps, true, verify_R_end, verify_Z_end)) return false;
        for (size_t block = 0; block < verify_indices.size(); ++block) {
            const int level_index = verify_indices[block];
            std::vector<double> distances;
            distances.reserve(theta_count);
            for (int theta_index = 0; theta_index < theta_count; ++theta_index) {
                const size_t index = block * theta_count + theta_index;
                double value, gR, gZ, gPhi;
                evaluate_psi_host(
                    psi, axis, config, nfp, verify_R_end[index], verify_Z_end[index], TWOPI / nfp,
                    value, gR, gZ, gPhi
                );
                const double gradient_norm = std::sqrt(gR * gR + gZ * gZ + std::pow(gPhi / verify_R_end[index], 2.0));
                distances.push_back(std::abs(value - screens[level_index].level) /
                                    std::max(gradient_norm, 1.0e-14));
            }
            screens[level_index].drift_p95 = percentile(distances, 0.95);
            screens[level_index].relative_drift_p95 = screens[level_index].drift_p95 /
                std::max(screens[level_index].radius_mean, 1.0e-14);
            screens[level_index].stable =
                screens[level_index].drift_p95 <= config.surface_drift_absolute_tolerance &&
                screens[level_index].relative_drift_p95 <= 0.30 &&
                screens[level_index].radius_max < config.surface_max_radius_scale * config.psi_a * 0.999;
            screens[level_index].strict = screens[level_index].stable &&
                screens[level_index].relative_drift_p95 <= config.surface_drift_relative_tolerance;
        }
    }
    return true;
}

bool validate_config(const SgpuScoreConfig& config, std::string& reason) {
    if (config.abi_version != SGPU_SCORE_ABI_VERSION || config.struct_size != sizeof(SgpuScoreConfig)) {
        reason = "score config ABI version or size mismatch";
        return false;
    }
    if (config.device_id < 0 || config.segments_per_coil <= 0 ||
        config.axis_grid < 8 || config.axis_fallback_grid < config.axis_grid ||
        config.axis_sample_count < 16 || config.psi_poly_degree < 2 ||
        config.psi_poly_degree > 24 || config.psi_m_tor < 0 || config.psi_m_tor > 32 ||
        config.surface_level_count <= 0 ||
        config.surface_level_count > SGPU_SCORE_MAX_SURFACE_LEVELS ||
        config.surface_theta_count < 16 || config.volume_point_count <= 0 ||
        config.alpha_fit_point_count <= 0 ||
        config.alpha_fit_point_count > config.volume_point_count) {
        reason = "invalid score configuration dimensions";
        return false;
    }
    return true;
}

void fill_early_components(
    const SgpuScoreConfig& config,
    const CoilMetrics& coil,
    const AxisData& axis,
    const PsiData& psi,
    const std::vector<SurfaceScreen>& screens,
    SgpuScoreResult& result
) {
    const double axis_residual = q_down(axis.selected.residual, config.score_axis_residual_scale, 0.8);
    const double topology = axis.selected.elliptic ? 1.0 : 0.1;
    const double aspect = q_down(std::max(axis.selected.ellipse_aspect - 1.0, 0.0), 1.0, 1.2, 0.8);
    result.components[SGPU_SCORE_COMPONENT_AXIS] = blend({{0.70, axis_residual}, {0.20, topology}, {0.10, aspect}});
    result.components[SGPU_SCORE_COMPONENT_PSI] = blend({
        {0.58, q_down(psi.angle_p95, config.score_psi_angle_p95_scale, 1.1)},
        {0.32, q_down(psi.angle_l2, config.score_psi_angle_l2_scale, 1.1)},
        {0.10, q_down(psi.train_rms, 5.0e-3, 1.0, 0.45)},
    });
    double minimum_drift = std::numeric_limits<double>::infinity();
    int strict_count = 0;
    for (const auto& screen : screens) {
        if (std::isfinite(screen.relative_drift_p95)) minimum_drift = std::min(minimum_drift, screen.relative_drift_p95);
        if (screen.strict) ++strict_count;
    }
    const double size = q_up(result.surface_inverse_aspect_ratio, config.score_surface_inverse_aspect_scale, 2.0);
    const double drift = q_down(minimum_drift, config.score_surface_drift_scale, 1.0, 0.15);
    const double count = q_up(strict_count, 2.0, 1.0);
    result.components[SGPU_SCORE_COMPONENT_SURFACE] = blend({{0.65, size}, {0.25, drift}, {0.10, count}});
    result.components[SGPU_SCORE_COMPONENT_COIL] = coil_component(coil);
}

bool run_downstream_gpu(
    void* field,
    const double* currents_a,
    int n_base_coils,
    int nfp,
    const AxisData& axis,
    const PsiData& psi,
    const SurfaceScreen& surface,
    const SgpuScoreConfig& config,
    SgpuScoreResult& result
) {
    (void)field;
    (void)currents_a;
    (void)n_base_coils;
    (void)nfp;
    (void)axis;
    (void)psi;
    (void)surface;
    (void)config;
    result.status = SGPU_SCORE_INTERNAL_ERROR;
    std::snprintf(
        result.error_message,
        sizeof(result.error_message),
        "%s",
        "native flux/alpha/QS stage is not linked yet"
    );
    sgpu_internal_set_error(result.error_message);
    return false;
}

}  // namespace

extern "C" {

int sgpu_default_score_config(SgpuScoreConfig* config) {
    if (!config) {
        sgpu_internal_set_error("score config pointer is null");
        return 1;
    }
    std::memset(config, 0, sizeof(*config));
    config->abi_version = SGPU_SCORE_ABI_VERSION;
    config->struct_size = sizeof(*config);
    config->device_id = 0;
    config->segments_per_coil = 256;
    config->target_M = 1;
    config->target_N = 0;
    config->axis_grid = 48;
    config->axis_fallback_grid = 96;
    config->axis_max_candidates = 16;
    config->axis_fallback_max_candidates = 96;
    config->axis_newton_iters = 6;
    config->axis_fallback_newton_iters = 8;
    config->axis_trace_steps = 960;
    config->axis_sample_count = 240;
    config->axis_span = 0.5;
    config->axis_tolerance = 1.0e-7;
    config->axis_r_floor = 1.0e-4;
    config->axis_fd_relative = 2.0e-4;
    config->axis_fd_absolute = 2.0e-6;
    config->axis_topology_margin = 2.0e-2;
    config->psi_poly_degree = 10;
    config->psi_m_tor = 12;
    config->psi_n_r = 80;
    config->psi_n_z = 80;
    config->psi_n_phi = 80;
    config->psi_validation_points = 4000;
    config->psi_a = 0.05;
    config->psi_rho_min = 0.002;
    config->psi_ridge = 1.0e-6;
    const double default_levels[] = {0.001, 0.002, 0.004, 0.008, 0.012, 0.02, 0.04, 0.08, 0.12, 0.16};
    config->surface_level_count = static_cast<int>(sizeof(default_levels) / sizeof(default_levels[0]));
    std::copy(default_levels, default_levels + config->surface_level_count, config->surface_levels);
    config->surface_theta_count = 256;
    config->surface_trace_steps = 800;
    config->surface_newton_iters = 20;
    config->surface_newton_tolerance = 1.0e-12;
    config->surface_max_radius_scale = 1.0;
    config->surface_drift_relative_tolerance = 0.05;
    config->surface_drift_absolute_tolerance = 5.0e-4;
    config->flux_level_count = 11;
    config->flux_phi_count = 8;
    config->flux_theta_count = 256;
    config->flux_radial_quadrature = 24;
    config->flux_polynomial_degree = 4;
    config->flux_boundary_tolerance = 1.0e-9;
    config->flux_section_relative_std_tolerance = 0.02;
    config->volume_point_count = 100000;
    config->volume_phi_count = 96;
    config->volume_theta_count = 32;
    config->alpha_fit_point_count = 30000;
    config->alpha_radial_order = 12;
    config->alpha_poloidal_order = 12;
    config->alpha_toroidal_order = 12;
    config->iota_degree = 0;
    config->radial_bin_count = 10;
    config->volume_rho_min = 0.08;
    config->alpha_ridge = 1.0e-7;
    const double weights[] = {18.0, 18.0, 18.0, 14.0, 20.0, 12.0};
    std::copy(weights, weights + SGPU_SCORE_COMPONENT_COUNT, config->score_weights);
    config->score_axis_residual_scale = 1.0e-5;
    config->score_psi_angle_p95_scale = 3.0e-3;
    config->score_psi_angle_l2_scale = 1.0e-3;
    config->score_surface_inverse_aspect_scale = 0.04;
    config->score_surface_drift_scale = 0.02;
    config->score_flux_section_std_scale = 0.01;
    config->score_flux_boundary_residual_scale = 1.0e-9;
    config->score_alpha_normal_B_scale = 1.0e-4;
    config->score_alpha_relative_l2_scale = 0.25;
    config->score_qs_global_scale = 0.05;
    config->score_qs_edge_scale = 0.07;
    sgpu_internal_set_error("");
    return 0;
}

int sgpu_score_coils(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig* config,
    SgpuScoreResult* result
) {
    if (!result) {
        sgpu_internal_set_error("score result pointer is null");
        return 1;
    }
    const int device_id = config ? config->device_id : -1;
    initialize_result(result, device_id);
    auto total_started = Clock::now();
    if (!coeffs_x || !coeffs_y || !coeffs_z || !currents_a || !config ||
        n_base_coils <= 0 || n_coeff < 3 || (n_coeff % 2) == 0 || nfp <= 0) {
        return fail_result(result, "invalid score input arrays or dimensions");
    }
    std::string config_error;
    if (!validate_config(*config, config_error)) return fail_result(result, config_error.c_str());
    if (cudaSetDevice(config->device_id) != cudaSuccess) return fail_result(result, "cudaSetDevice failed");

    auto stage_started = Clock::now();
    const CoilMetrics coil = compute_coil_metrics(
        coeffs_x, coeffs_y, coeffs_z, currents_a, n_base_coils, n_coeff, nfp
    );
    result->timings[SGPU_SCORE_TIME_COIL_GEOMETRY] = seconds_since(stage_started);
    result->coil_length_mean = coil.length_mean;
    result->coil_curvature_p95 = coil.curvature_p95;
    result->coil_curvature_max = coil.curvature_max;
    result->coil_min_intercoil_distance = coil.min_intercoil_distance;
    result->coil_min_axis_distance = coil.min_axis_distance;
    result->coil_high_mode_energy_fraction = coil.high_mode_fraction;
    result->coil_current_abs_max_a = coil.current_abs_max;

    void* field = nullptr;
    stage_started = Clock::now();
    if (sgpu_create_field(
            coeffs_x, coeffs_y, coeffs_z, currents_a, n_base_coils, n_coeff,
            nfp, config->segments_per_coil, config->device_id, &field)) {
        return fail_from_backend(result, "field creation");
    }
    result->timings[SGPU_SCORE_TIME_FIELD_CREATE] = seconds_since(stage_started);
    result->stage_completed = SCORE_STAGE_FIELD;

    int return_code = 0;
    do {
        AxisData axis;
        stage_started = Clock::now();
        if (!find_axis_native(
                field, coeffs_x, coeffs_y, coeffs_z, n_base_coils, n_coeff,
                nfp, *config, axis)) {
            return_code = fail_from_backend(result, "axis search");
            break;
        }
        result->timings[SGPU_SCORE_TIME_AXIS_SEARCH] = seconds_since(stage_started);
        result->axis_candidate_count = axis.candidate_count;
        if (axis.R.empty()) {
            result->status = SGPU_SCORE_NO_AXIS;
            result->stage_completed = SCORE_STAGE_AXIS;
            result->components[SGPU_SCORE_COMPONENT_COIL] = coil_component(coil);
            break;
        }
        result->axis_R = axis.selected.R;
        result->axis_Z = axis.selected.Z;
        result->axis_residual = axis.selected.residual;
        result->axis_topology_trace = axis.selected.topology_trace;
        result->axis_topology_det = axis.selected.topology_det;
        result->axis_ellipse_aspect = axis.selected.ellipse_aspect;
        result->stage_completed = SCORE_STAGE_AXIS;

        PsiData psi;
        if (!fit_psi_native(
                field, axis, nfp, *config, psi,
                result->timings[SGPU_SCORE_TIME_PSI_POINTS],
                result->timings[SGPU_SCORE_TIME_PSI_FIT])) {
            return_code = fail_from_backend(result, "psi fit");
            break;
        }
        stage_started = Clock::now();
        if (!validate_psi_native(field, axis, nfp, *config, psi)) {
            return_code = fail_from_backend(result, "psi validation");
            break;
        }
        result->timings[SGPU_SCORE_TIME_PSI_VALIDATE] = seconds_since(stage_started);
        result->psi_train_rms = psi.train_rms;
        result->psi_angle_mean = psi.angle_mean;
        result->psi_angle_p95 = psi.angle_p95;
        result->psi_angle_l2 = psi.angle_l2;
        result->stage_completed = SCORE_STAGE_PSI;

        std::vector<SurfaceScreen> screens;
        stage_started = Clock::now();
        if (!screen_surfaces_native(field, axis, psi, nfp, *config, screens)) {
            return_code = fail_from_backend(result, "surface screen");
            break;
        }
        result->timings[SGPU_SCORE_TIME_SURFACE_SCREEN] = seconds_since(stage_started);
        const auto best_strict = std::max_element(screens.begin(), screens.end(), [](const auto& lhs, const auto& rhs) {
            return (lhs.strict ? lhs.level : -1.0) < (rhs.strict ? rhs.level : -1.0);
        });
        result->stable_surface_count = static_cast<int>(std::count_if(screens.begin(), screens.end(), [](const auto& screen) {
            return screen.stable;
        }));
        result->stage_completed = SCORE_STAGE_SURFACE;
        if (result->stable_surface_count == 0) {
            result->status = SGPU_SCORE_NO_SURFACE;
            fill_early_components(*config, coil, axis, psi, screens, *result);
            break;
        }
        if (best_strict == screens.end() || !best_strict->strict) {
            result->status = SGPU_SCORE_DRIFT_REJECTED;
            fill_early_components(*config, coil, axis, psi, screens, *result);
            break;
        }
        result->surface_level = best_strict->level;
        result->surface_drift_relative_p95 = best_strict->relative_drift_p95;
        if (!run_downstream_gpu(
                field, currents_a, n_base_coils, nfp, axis, psi,
                *best_strict, *config, *result)) {
            if (result->status == SGPU_SCORE_INTERNAL_ERROR) return_code = 1;
            break;
        }
        fill_early_components(*config, coil, axis, psi, screens, *result);
    } while (false);

    sgpu_destroy_field(field);
    result->timings[SGPU_SCORE_TIME_TOTAL] = seconds_since(total_started);
    return return_code;
}

}  // extern "C"
