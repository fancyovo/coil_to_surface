#include "coil_field.h"
#include "coil_field_internal.h"

#include <cublas_v2.h>
#include <cub/cub.cuh>
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

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

void finalize_score(const SgpuScoreConfig& config, SgpuScoreResult& result) {
    if (result.status == SGPU_SCORE_INTERNAL_ERROR) return;
    if (result.stage_completed < SCORE_STAGE_FLUX) {
        result.components[SGPU_SCORE_COMPONENT_COORDINATE] = 0.08;
    }
    if (result.stage_completed < SCORE_STAGE_QS) {
        result.components[SGPU_SCORE_COMPONENT_VOLUME_QS] = 0.04;
    }
    double total_weight = 0.0;
    double weighted_score = 0.0;
    for (int component = 0; component < SGPU_SCORE_COMPONENT_COUNT; ++component) {
        const double value = clip01(result.components[component]);
        total_weight += std::max(config.score_weights[component], 0.0);
        weighted_score += std::max(config.score_weights[component], 0.0) * value;
        result.components[component] = 100.0 * value;
    }
    result.score = total_weight > 0.0 ? 100.0 * clip01(weighted_score / total_weight) : 0.0;
    if (result.status == SGPU_SCORE_OK) result.stage_completed = SCORE_STAGE_COMPLETE;
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { allocate(count); }
    ~DeviceBuffer() { cudaFree(pointer_); }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept : pointer_(other.pointer_), count_(other.count_) {
        other.pointer_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            cudaFree(pointer_);
            pointer_ = other.pointer_;
            count_ = other.count_;
            other.pointer_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    bool allocate(size_t count) {
        cudaFree(pointer_);
        pointer_ = nullptr;
        count_ = 0;
        if (count == 0) return true;
        if (cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)) != cudaSuccess) return false;
        count_ = count;
        return true;
    }

    T* data() { return pointer_; }
    const T* data() const { return pointer_; }
    size_t size() const { return count_; }

private:
    T* pointer_ = nullptr;
    size_t count_ = 0;
};

bool cuda_stage_ok(cudaError_t error, SgpuScoreResult& result, const char* stage) {
    if (error == cudaSuccess) return true;
    std::string message = stage ? stage : "CUDA stage";
    message += ": ";
    message += cudaGetErrorString(error);
    fail_result(&result, message.c_str());
    return false;
}

__device__ inline double atomic_add_double(double* address, double value) {
#if __CUDA_ARCH__ >= 600
    return atomicAdd(address, value);
#else
    auto* integer_address = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *integer_address;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(
            integer_address,
            assumed,
            __double_as_longlong(value + __longlong_as_double(assumed))
        );
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

template <typename T>
bool copy_to_device(DeviceBuffer<T>& destination, const std::vector<T>& source) {
    return destination.allocate(source.size()) &&
           cudaMemcpy(destination.data(), source.data(), source.size() * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
}

struct DevicePsiData {
    DeviceBuffer<float> coeffs;
    DeviceBuffer<int> mode_a;
    DeviceBuffer<int> mode_b;
    DeviceBuffer<int> mode_m;
    DeviceBuffer<int> mode_kind;
    DeviceBuffer<float> axis_R;
    DeviceBuffer<float> axis_Z;
    DeviceBuffer<float> axis_R_phi;
    DeviceBuffer<float> axis_Z_phi;
};

bool upload_psi_data(
    const PsiData& psi,
    const AxisData& axis,
    DevicePsiData& device
) {
    std::vector<float> coefficients(psi.coeffs.begin(), psi.coeffs.end());
    std::vector<float> axis_R(axis.R.begin(), axis.R.end());
    std::vector<float> axis_Z(axis.Z.begin(), axis.Z.end());
    std::vector<float> axis_R_phi(axis.R_phi.begin(), axis.R_phi.end());
    std::vector<float> axis_Z_phi(axis.Z_phi.begin(), axis.Z_phi.end());
    return copy_to_device(device.coeffs, coefficients) &&
           copy_to_device(device.mode_a, psi.modes.a) &&
           copy_to_device(device.mode_b, psi.modes.b) &&
           copy_to_device(device.mode_m, psi.modes.m) &&
           copy_to_device(device.mode_kind, psi.modes.kind) &&
           copy_to_device(device.axis_R, axis_R) &&
           copy_to_device(device.axis_Z, axis_Z) &&
           copy_to_device(device.axis_R_phi, axis_R_phi) &&
           copy_to_device(device.axis_Z_phi, axis_Z_phi);
}

__device__ inline void periodic_hermite_device(
    float phi,
    const float* values,
    const float* derivatives,
    int count,
    float period,
    float& value,
    float& derivative
) {
    float wrapped = fmodf(phi, period);
    if (wrapped < 0.0f) wrapped += period;
    const float position = wrapped * count / period;
    int index0 = static_cast<int>(floorf(position));
    if (index0 >= count) index0 = 0;
    const int index1 = index0 + 1 == count ? 0 : index0 + 1;
    const float t = position - index0;
    const float h = period / count;
    const float h00 = (2.0f * t - 3.0f) * t * t + 1.0f;
    const float h10 = ((t - 2.0f) * t + 1.0f) * t;
    const float h01 = (-2.0f * t + 3.0f) * t * t;
    const float h11 = (t - 1.0f) * t * t;
    value = h00 * values[index0] + h10 * h * derivatives[index0] +
            h01 * values[index1] + h11 * h * derivatives[index1];
    const float dh00 = (6.0f * t - 6.0f) * t;
    const float dh10 = (3.0f * t - 4.0f) * t + 1.0f;
    const float dh01 = (-6.0f * t + 6.0f) * t;
    const float dh11 = (3.0f * t - 2.0f) * t;
    derivative = (dh00 * values[index0] + dh01 * values[index1]) / h +
                 dh10 * derivatives[index0] + dh11 * derivatives[index1];
}

__device__ inline void evaluate_psi_device(
    const float* coeffs,
    const int* mode_a,
    const int* mode_b,
    const int* mode_m,
    const int* mode_kind,
    int n_coeff,
    int nfp,
    int degree,
    int m_tor,
    float a_scale,
    float R,
    float Z,
    float phi,
    float axis_R,
    float axis_Z,
    float axis_R_phi,
    float axis_Z_phi,
    float& value,
    float& grad_R,
    float& grad_Z,
    float& grad_phi
) {
    float xpow[25];
    float zpow[25];
    float cosv[33];
    float sinv[33];
    const float X = (R - axis_R) / a_scale;
    const float Y = (Z - axis_Z) / a_scale;
    xpow[0] = 1.0f;
    zpow[0] = 1.0f;
    for (int power = 1; power <= degree; ++power) {
        xpow[power] = xpow[power - 1] * X;
        zpow[power] = zpow[power - 1] * Y;
    }
    cosv[0] = 1.0f;
    sinv[0] = 0.0f;
    for (int mode = 1; mode <= m_tor; ++mode) {
        sincosf(static_cast<float>(mode * nfp) * phi, &sinv[mode], &cosv[mode]);
    }
    value = X * X;
    grad_R = 2.0f * X / a_scale;
    grad_Z = 0.0f;
    grad_phi = -axis_R_phi * grad_R;
    for (int index = 0; index < n_coeff; ++index) {
        const int ax = mode_a[index];
        const int bz = mode_b[index];
        const int mode = mode_m[index];
        const float trig = mode == 0 ? 1.0f : (mode_kind[index] == 0 ? cosv[mode] : sinv[mode]);
        const float trig_phi = mode == 0 ? 0.0f : static_cast<float>(mode * nfp) *
            (mode_kind[index] == 0 ? -sinv[mode] : cosv[mode]);
        const float mono = xpow[ax] * zpow[bz];
        const float derivative_x = ax > 0 ? ax * xpow[ax - 1] * zpow[bz] / a_scale : 0.0f;
        const float derivative_z = bz > 0 ? bz * xpow[ax] * zpow[bz - 1] / a_scale : 0.0f;
        const float coefficient = coeffs[index];
        value += coefficient * mono * trig;
        grad_R += coefficient * derivative_x * trig;
        grad_Z += coefficient * derivative_z * trig;
        grad_phi += coefficient *
            (mono * trig_phi - axis_R_phi * derivative_x * trig - axis_Z_phi * derivative_z * trig);
    }
}

__global__ void build_ray_coefficients_kernel(
    const float* __restrict__ coeffs,
    const int* __restrict__ mode_a,
    const int* __restrict__ mode_b,
    const int* __restrict__ mode_m,
    const int* __restrict__ mode_kind,
    int n_coeff,
    int nfp,
    int degree,
    int m_tor,
    int n_phi,
    int n_theta,
    float phi_offset,
    float theta_offset,
    float* __restrict__ ray_coefficients
) {
    const int ray = blockIdx.x * blockDim.x + threadIdx.x;
    const int ray_count = n_phi * n_theta;
    if (ray >= ray_count) return;
    const int phi_index = ray / n_theta;
    const int theta_index = ray - phi_index * n_theta;
    const float phi = (phi_index + phi_offset) * static_cast<float>(TWOPI) /
                      static_cast<float>(nfp * n_phi);
    float shifted_theta = theta_index + theta_offset + 0.217f * phi_index;
    shifted_theta -= floorf(shifted_theta / n_theta) * n_theta;
    const float theta = shifted_theta * static_cast<float>(TWOPI) / n_theta;
    float cosine, sine;
    sincosf(theta, &sine, &cosine);
    float cpow[25];
    float spow[25];
    float cosv[33];
    float sinv[33];
    cpow[0] = 1.0f;
    spow[0] = 1.0f;
    for (int power = 1; power <= degree; ++power) {
        cpow[power] = cpow[power - 1] * cosine;
        spow[power] = spow[power - 1] * sine;
    }
    cosv[0] = 1.0f;
    sinv[0] = 0.0f;
    for (int mode = 1; mode <= m_tor; ++mode) {
        sincosf(static_cast<float>(mode * nfp) * phi, &sinv[mode], &cosv[mode]);
    }
    float local[25];
    for (int power = 0; power <= degree; ++power) local[power] = 0.0f;
    local[2] = cosine * cosine;
    for (int index = 0; index < n_coeff; ++index) {
        const int mode = mode_m[index];
        const float trig = mode == 0 ? 1.0f : (mode_kind[index] == 0 ? cosv[mode] : sinv[mode]);
        const int total = mode_a[index] + mode_b[index];
        local[total] += coeffs[index] * cpow[mode_a[index]] * spow[mode_b[index]] * trig;
    }
    for (int power = 0; power <= degree; ++power) {
        ray_coefficients[static_cast<size_t>(ray) * (degree + 1) + power] = local[power];
    }
}

__global__ void solve_boundary_radii_kernel(
    const float* __restrict__ ray_coefficients,
    int ray_count,
    int degree,
    float a_scale,
    const float* __restrict__ levels,
    int level_count,
    int iterations,
    float tolerance,
    float max_radius_scale,
    float* __restrict__ radii,
    float* __restrict__ residuals
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = ray_count * level_count;
    if (index >= total) return;
    const int level_index = index / ray_count;
    const int ray = index - level_index * ray_count;
    const float* polynomial = ray_coefficients + static_cast<size_t>(ray) * (degree + 1);
    const float level = levels[level_index];
    const float q = polynomial[2];
    float radius = q > 1.0e-10f
        ? a_scale * sqrtf(fmaxf(level, 0.0f) / q)
        : a_scale * sqrtf(fmaxf(level, 1.0e-16f));
    const float maximum = max_radius_scale * a_scale;
    radius = fminf(maximum, fmaxf(1.0e-12f * a_scale, radius));
    float final_residual = INFINITY;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const float u = radius / a_scale;
        float value = 0.0f;
        float derivative = 0.0f;
        float previous = u;
        float power = u * u;
        for (int degree_now = 2; degree_now <= degree; ++degree_now) {
            if (degree_now > 2) {
                previous = power;
                power *= u;
            }
            value += polynomial[degree_now] * power;
            derivative += degree_now * polynomial[degree_now] * previous / a_scale;
        }
        final_residual = value - level;
        if (fabsf(final_residual) <= tolerance) break;
        const float denominator = fabsf(derivative) > 1.0e-14f
            ? derivative : copysignf(1.0e-14f, derivative == 0.0f ? 1.0f : derivative);
        const float limit = 0.4f * fmaxf(radius, 1.0e-10f);
        const float step = fminf(limit, fmaxf(-limit, final_residual / denominator));
        radius = fminf(maximum, fmaxf(1.0e-12f * a_scale, radius - step));
    }
    radii[index] = radius;
    residuals[index] = fabsf(final_residual);
}

void gauss_legendre(int count, std::vector<float>& nodes, std::vector<float>& weights) {
    nodes.resize(count);
    weights.resize(count);
    const int half = (count + 1) / 2;
    for (int index = 0; index < half; ++index) {
        double root = std::cos(PI * (index + 0.75) / (count + 0.5));
        double derivative = 0.0;
        for (int iteration = 0; iteration < 32; ++iteration) {
            double p0 = 1.0;
            double p1 = root;
            for (int degree = 2; degree <= count; ++degree) {
                const double next = ((2.0 * degree - 1.0) * root * p1 - (degree - 1.0) * p0) / degree;
                p0 = p1;
                p1 = next;
            }
            derivative = count * (root * p1 - p0) / (root * root - 1.0);
            const double step = p1 / derivative;
            root -= step;
            if (std::abs(step) < 1.0e-15) break;
        }
        const double weight = 2.0 / ((1.0 - root * root) * derivative * derivative);
        nodes[index] = static_cast<float>(-root);
        nodes[count - 1 - index] = static_cast<float>(root);
        weights[index] = static_cast<float>(weight);
        weights[count - 1 - index] = static_cast<float>(weight);
    }
}

__global__ void generate_flux_points_kernel(
    const float* __restrict__ boundary_radii,
    const float* __restrict__ gauss_nodes,
    int n_phi,
    int n_levels,
    int n_theta,
    int n_radial,
    int nfp,
    const float* __restrict__ axis_R,
    const float* __restrict__ axis_Z,
    const float* __restrict__ axis_R_phi,
    const float* __restrict__ axis_Z_phi,
    int axis_count,
    float* __restrict__ xyz
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_phi * n_levels * n_theta * n_radial;
    if (index >= total) return;
    int remainder = index;
    const int radial_index = remainder % n_radial;
    remainder /= n_radial;
    const int theta_index = remainder % n_theta;
    remainder /= n_theta;
    const int level_index = remainder % n_levels;
    const int phi_index = remainder / n_levels;
    const int ray = phi_index * n_theta + theta_index;
    const float boundary = boundary_radii[level_index * (n_phi * n_theta) + ray];
    const float radius = 0.5f * boundary * (gauss_nodes[radial_index] + 1.0f);
    const float phi = phi_index * static_cast<float>(TWOPI) / static_cast<float>(nfp * n_phi);
    const float theta = theta_index * static_cast<float>(TWOPI) / n_theta;
    float axis_radius, axis_radius_phi, axis_z, axis_z_phi;
    periodic_hermite_device(
        phi, axis_R, axis_R_phi, axis_count, static_cast<float>(TWOPI) / nfp,
        axis_radius, axis_radius_phi
    );
    periodic_hermite_device(
        phi, axis_Z, axis_Z_phi, axis_count, static_cast<float>(TWOPI) / nfp,
        axis_z, axis_z_phi
    );
    float sine_phi, cosine_phi, sine_theta, cosine_theta;
    sincosf(phi, &sine_phi, &cosine_phi);
    sincosf(theta, &sine_theta, &cosine_theta);
    const float R = axis_radius + radius * cosine_theta;
    xyz[3 * index] = R * cosine_phi;
    xyz[3 * index + 1] = R * sine_phi;
    xyz[3 * index + 2] = axis_z + radius * sine_theta;
}

__global__ void reduce_flux_sections_kernel(
    const float* __restrict__ B,
    const float* __restrict__ boundary_radii,
    const float* __restrict__ gauss_nodes,
    const float* __restrict__ gauss_weights,
    int n_phi,
    int n_levels,
    int n_theta,
    int n_radial,
    int nfp,
    double* __restrict__ sections
) {
    extern __shared__ double shared[];
    const int section = blockIdx.x;
    const int level_index = section % n_levels;
    const int phi_index = section / n_levels;
    const float phi = phi_index * static_cast<float>(TWOPI) / static_cast<float>(nfp * n_phi);
    float sine_phi, cosine_phi;
    sincosf(phi, &sine_phi, &cosine_phi);
    double sum = 0.0;
    const int per_section = n_theta * n_radial;
    for (int local = threadIdx.x; local < per_section; local += blockDim.x) {
        const int radial_index = local % n_radial;
        const int theta_index = local / n_radial;
        const int ray = phi_index * n_theta + theta_index;
        const float boundary = boundary_radii[level_index * (n_phi * n_theta) + ray];
        const float radius = 0.5f * boundary * (gauss_nodes[radial_index] + 1.0f);
        const float radial_weight = 0.5f * boundary * gauss_weights[radial_index];
        const int point = ((phi_index * n_levels + level_index) * n_theta + theta_index) * n_radial + radial_index;
        const float B_phi = -B[3 * point] * sine_phi + B[3 * point + 1] * cosine_phi;
        sum += static_cast<double>(B_phi) * radius * radial_weight;
    }
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) sections[section] = shared[0] / n_theta;
}

bool solve_small_least_squares(
    const std::vector<double>& matrix,
    const std::vector<double>& rhs,
    int rows,
    int columns,
    std::vector<double>& solution
) {
    std::vector<double> q(matrix);
    std::vector<double> r(static_cast<size_t>(columns) * columns, 0.0);
    for (int column = 0; column < columns; ++column) {
        for (int previous = 0; previous < column; ++previous) {
            double dot = 0.0;
            for (int row = 0; row < rows; ++row) {
                dot += q[static_cast<size_t>(row) * columns + previous] *
                       q[static_cast<size_t>(row) * columns + column];
            }
            r[static_cast<size_t>(previous) * columns + column] = dot;
            for (int row = 0; row < rows; ++row) {
                q[static_cast<size_t>(row) * columns + column] -=
                    dot * q[static_cast<size_t>(row) * columns + previous];
            }
        }
        double norm2 = 0.0;
        for (int row = 0; row < rows; ++row) {
            const double value = q[static_cast<size_t>(row) * columns + column];
            norm2 += value * value;
        }
        const double norm = std::sqrt(norm2);
        if (!(norm > 1.0e-20)) return false;
        r[static_cast<size_t>(column) * columns + column] = norm;
        for (int row = 0; row < rows; ++row) q[static_cast<size_t>(row) * columns + column] /= norm;
    }
    std::vector<double> qtb(columns, 0.0);
    for (int column = 0; column < columns; ++column) {
        for (int row = 0; row < rows; ++row) {
            qtb[column] += q[static_cast<size_t>(row) * columns + column] * rhs[row];
        }
    }
    solution.assign(columns, 0.0);
    for (int row = columns - 1; row >= 0; --row) {
        double value = qtb[row];
        for (int column = row + 1; column < columns; ++column) {
            value -= r[static_cast<size_t>(row) * columns + column] * solution[column];
        }
        solution[row] = value / r[static_cast<size_t>(row) * columns + row];
    }
    return true;
}

struct FluxCalibrationNative {
    double edge_level = 0.0;
    double edge_flux = 0.0;
    double fit_relative_rms = 0.0;
    double section_relative_std_edge = 0.0;
    double boundary_residual_max = 0.0;
    double derivative_min = 0.0;
    double derivative_max = 0.0;
    double volume = 0.0;
    double major_radius = 0.0;
    double effective_minor_radius = 0.0;
    std::vector<float> coefficients;
};

bool calibrate_flux_native(
    void* field,
    const DevicePsiData& device_psi,
    const AxisData& axis,
    const PsiData& psi,
    int nfp,
    double edge_level,
    const SgpuScoreConfig& config,
    FluxCalibrationNative& calibration,
    SgpuScoreResult& result
) {
    const int n_phi = config.flux_phi_count;
    const int n_levels = config.flux_level_count;
    const int n_theta = config.flux_theta_count;
    const int n_radial = config.flux_radial_quadrature;
    const int ray_count = n_phi * n_theta;
    const int field_points = n_phi * n_levels * n_theta * n_radial;
    std::vector<float> levels(n_levels);
    for (int index = 0; index < n_levels; ++index) {
        const double fraction = static_cast<double>(index + 1) / n_levels;
        levels[index] = static_cast<float>(edge_level * fraction * fraction);
    }
    std::vector<float> gauss_nodes, gauss_weights;
    gauss_legendre(n_radial, gauss_nodes, gauss_weights);
    DeviceBuffer<float> d_levels, d_gauss_nodes, d_gauss_weights;
    DeviceBuffer<float> d_ray_coefficients(static_cast<size_t>(ray_count) * (config.psi_poly_degree + 1));
    DeviceBuffer<float> d_radii(static_cast<size_t>(ray_count) * n_levels);
    DeviceBuffer<float> d_residuals(d_radii.size());
    DeviceBuffer<float> d_xyz(static_cast<size_t>(field_points) * 3);
    DeviceBuffer<float> d_B(d_xyz.size());
    DeviceBuffer<double> d_sections(static_cast<size_t>(n_phi) * n_levels);
    if (!copy_to_device(d_levels, levels) || !copy_to_device(d_gauss_nodes, gauss_nodes) ||
        !copy_to_device(d_gauss_weights, gauss_weights) || !d_ray_coefficients.data() ||
        !d_radii.data() || !d_residuals.data() || !d_xyz.data() || !d_B.data() || !d_sections.data()) {
        fail_result(&result, "flux device allocation failed");
        return false;
    }
    constexpr int threads = 256;
    build_ray_coefficients_kernel<<<(ray_count + threads - 1) / threads, threads>>>(
        device_psi.coeffs.data(), device_psi.mode_a.data(), device_psi.mode_b.data(),
        device_psi.mode_m.data(), device_psi.mode_kind.data(), static_cast<int>(psi.coeffs.size()),
        nfp, config.psi_poly_degree, config.psi_m_tor, n_phi, n_theta, 0.0f, 0.0f,
        d_ray_coefficients.data()
    );
    const int radius_count = ray_count * n_levels;
    solve_boundary_radii_kernel<<<(radius_count + threads - 1) / threads, threads>>>(
        d_ray_coefficients.data(), ray_count, config.psi_poly_degree,
        static_cast<float>(config.psi_a), d_levels.data(), n_levels,
        config.surface_newton_iters, static_cast<float>(config.surface_newton_tolerance),
        static_cast<float>(config.surface_max_radius_scale), d_radii.data(), d_residuals.data()
    );
    generate_flux_points_kernel<<<(field_points + threads - 1) / threads, threads>>>(
        d_radii.data(), d_gauss_nodes.data(), n_phi, n_levels, n_theta, n_radial, nfp,
        device_psi.axis_R.data(), device_psi.axis_Z.data(),
        device_psi.axis_R_phi.data(), device_psi.axis_Z_phi.data(),
        static_cast<int>(axis.R.size()), d_xyz.data()
    );
    if (!cuda_stage_ok(cudaGetLastError(), result, "flux geometry kernels") ||
        sgpu_internal_eval_B_f32_device(field, d_xyz.data(), d_B.data(), field_points)) {
        if (result.status != SGPU_SCORE_INTERNAL_ERROR) fail_from_backend(&result, "flux field evaluation");
        return false;
    }
    reduce_flux_sections_kernel<<<n_phi * n_levels, threads, threads * sizeof(double)>>>(
        d_B.data(), d_radii.data(), d_gauss_nodes.data(), d_gauss_weights.data(),
        n_phi, n_levels, n_theta, n_radial, nfp, d_sections.data()
    );
    if (!cuda_stage_ok(cudaDeviceSynchronize(), result, "flux reduction")) return false;

    std::vector<float> radii(d_radii.size()), residuals(d_residuals.size());
    std::vector<double> sections(d_sections.size());
    if (!cuda_stage_ok(cudaMemcpy(radii.data(), d_radii.data(), radii.size() * sizeof(float), cudaMemcpyDeviceToHost), result, "flux copy radii") ||
        !cuda_stage_ok(cudaMemcpy(residuals.data(), d_residuals.data(), residuals.size() * sizeof(float), cudaMemcpyDeviceToHost), result, "flux copy residuals") ||
        !cuda_stage_ok(cudaMemcpy(sections.data(), d_sections.data(), sections.size() * sizeof(double), cudaMemcpyDeviceToHost), result, "flux copy sections")) {
        return false;
    }
    calibration.boundary_residual_max = *std::max_element(residuals.begin(), residuals.end());
    std::vector<double> mean(n_levels, 0.0), section_std(n_levels, 0.0);
    for (int level = 0; level < n_levels; ++level) {
        for (int phi_index = 0; phi_index < n_phi; ++phi_index) {
            mean[level] += sections[static_cast<size_t>(phi_index) * n_levels + level] / n_phi;
        }
        for (int phi_index = 0; phi_index < n_phi; ++phi_index) {
            const double delta = sections[static_cast<size_t>(phi_index) * n_levels + level] - mean[level];
            section_std[level] += delta * delta / n_phi;
        }
        section_std[level] = std::sqrt(section_std[level]);
    }
    calibration.section_relative_std_edge = section_std.back() / std::max(std::abs(mean.back()), 1.0e-30);
    const int polynomial_degree = config.flux_polynomial_degree;
    std::vector<double> matrix(static_cast<size_t>(n_levels) * polynomial_degree);
    for (int row = 0; row < n_levels; ++row) {
        double power = levels[row];
        for (int column = 0; column < polynomial_degree; ++column) {
            matrix[static_cast<size_t>(row) * polynomial_degree + column] = power;
            power *= levels[row];
        }
    }
    std::vector<double> coefficients;
    if (!solve_small_least_squares(matrix, mean, n_levels, polynomial_degree, coefficients)) {
        fail_result(&result, "flux calibration linear least squares failed");
        return false;
    }
    calibration.coefficients.assign(coefficients.begin(), coefficients.end());
    double fit_error2 = 0.0;
    calibration.derivative_min = std::numeric_limits<double>::infinity();
    calibration.derivative_max = -std::numeric_limits<double>::infinity();
    for (int row = 0; row < n_levels; ++row) {
        double fitted = 0.0;
        double power = levels[row];
        for (int column = 0; column < polynomial_degree; ++column) {
            fitted += coefficients[column] * power;
            power *= levels[row];
        }
        const double delta = fitted - mean[row];
        fit_error2 += delta * delta / n_levels;
    }
    for (int index = 0; index <= 200; ++index) {
        const double level = edge_level * index / 200.0;
        double derivative = 0.0;
        double power = 1.0;
        for (int column = 0; column < polynomial_degree; ++column) {
            derivative += (column + 1) * coefficients[column] * power;
            power *= level;
        }
        calibration.derivative_min = std::min(calibration.derivative_min, derivative);
        calibration.derivative_max = std::max(calibration.derivative_max, derivative);
    }
    calibration.edge_level = edge_level;
    calibration.edge_flux = 0.0;
    double edge_power = edge_level;
    for (double coefficient : coefficients) {
        calibration.edge_flux += coefficient * edge_power;
        edge_power *= edge_level;
    }
    calibration.fit_relative_rms = std::sqrt(fit_error2) / std::max(std::abs(mean.back()), 1.0e-30);
    double volume_integrand_sum = 0.0;
    double major_radius_sum = 0.0;
    for (int phi_index = 0; phi_index < n_phi; ++phi_index) {
        const double phi = TWOPI * phi_index / static_cast<double>(nfp * n_phi);
        double axis_R, axis_R_phi, axis_Z, axis_Z_phi;
        periodic_hermite_host(phi, axis.R, axis.R_phi, nfp, axis_R, axis_R_phi);
        periodic_hermite_host(phi, axis.Z, axis.Z_phi, nfp, axis_Z, axis_Z_phi);
        major_radius_sum += axis_R / n_phi;
        for (int theta_index = 0; theta_index < n_theta; ++theta_index) {
            const size_t ray = static_cast<size_t>(phi_index) * n_theta + theta_index;
            const double radius = radii[static_cast<size_t>(n_levels - 1) * ray_count + ray];
            const double theta = TWOPI * theta_index / n_theta;
            volume_integrand_sum += (0.5 * axis_R * radius * radius +
                                     std::cos(theta) * radius * radius * radius / 3.0) /
                                    static_cast<double>(n_phi * n_theta);
        }
    }
    calibration.volume = TWOPI * TWOPI * volume_integrand_sum;
    calibration.major_radius = major_radius_sum;
    calibration.effective_minor_radius = std::sqrt(
        std::max(calibration.volume, 0.0) /
        std::max(2.0 * PI * PI * calibration.major_radius, 1.0e-30)
    );
    const bool monotone = calibration.edge_flux >= 0.0
        ? calibration.derivative_min > 0.0
        : calibration.derivative_max < 0.0;
    return calibration.boundary_residual_max <= config.flux_boundary_tolerance &&
           calibration.section_relative_std_edge <= config.flux_section_relative_std_tolerance &&
           monotone;
}

struct DeviceVolumePoints {
    DeviceBuffer<float> xyz;
    DeviceBuffer<float> grad_s;
    DeviceBuffer<float> grad_theta;
    DeviceBuffer<float> grad_phi;
    DeviceBuffer<float> s;
    DeviceBuffer<float> rho;
    DeviceBuffer<float> theta;
    DeviceBuffer<float> phi;
    DeviceBuffer<float> R;
    DeviceBuffer<float> volume_weight;
    DeviceBuffer<float> flux_derivative;
    int count = 0;

    bool allocate(int point_count) {
        count = point_count;
        return xyz.allocate(static_cast<size_t>(point_count) * 3) &&
               grad_s.allocate(static_cast<size_t>(point_count) * 3) &&
               grad_theta.allocate(static_cast<size_t>(point_count) * 3) &&
               grad_phi.allocate(static_cast<size_t>(point_count) * 3) &&
               s.allocate(point_count) && rho.allocate(point_count) &&
               theta.allocate(point_count) && phi.allocate(point_count) &&
               R.allocate(point_count) && volume_weight.allocate(point_count) &&
               flux_derivative.allocate(point_count);
    }
};

__global__ void generate_volume_candidates_kernel(
    const float* __restrict__ boundary_radii,
    int n_phi,
    int n_theta,
    int n_radial,
    int nfp,
    float rho_min,
    float edge_level,
    const float* __restrict__ flux_coefficients,
    int flux_degree,
    float edge_flux,
    const float* __restrict__ psi_coefficients,
    const int* __restrict__ mode_a,
    const int* __restrict__ mode_b,
    const int* __restrict__ mode_m,
    const int* __restrict__ mode_kind,
    int psi_coefficient_count,
    int psi_degree,
    int psi_m_tor,
    float psi_a,
    const float* __restrict__ axis_R,
    const float* __restrict__ axis_Z,
    const float* __restrict__ axis_R_phi,
    const float* __restrict__ axis_Z_phi,
    int axis_count,
    float* __restrict__ xyz,
    float* __restrict__ grad_s,
    float* __restrict__ grad_theta,
    float* __restrict__ grad_phi,
    float* __restrict__ s_out,
    float* __restrict__ rho_out,
    float* __restrict__ theta_out,
    float* __restrict__ phi_out,
    float* __restrict__ R_out,
    float* __restrict__ volume_weight,
    float* __restrict__ flux_derivative_out,
    unsigned char* __restrict__ valid,
    int* __restrict__ source_indices
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_phi * n_theta * n_radial;
    if (index >= total) return;
    int remainder = index;
    const int radial_index = remainder % n_radial;
    remainder /= n_radial;
    const int theta_index = remainder % n_theta;
    const int phi_index = remainder / n_theta;
    const int ray = phi_index * n_theta + theta_index;
    const float boundary = boundary_radii[ray];
    const float radial_fraction = sqrtf(
        rho_min * rho_min + (1.0f - rho_min * rho_min) *
        (radial_index + 0.371f) / n_radial
    );
    const float radius = boundary * radial_fraction;
    const float phi = (phi_index + 0.417f) * static_cast<float>(TWOPI) /
                      static_cast<float>(nfp * n_phi);
    float shifted_theta = theta_index + 0.613f + 0.217f * phi_index;
    shifted_theta -= floorf(shifted_theta / n_theta) * n_theta;
    const float geometric_theta = shifted_theta * static_cast<float>(TWOPI) / n_theta;
    float axis_radius, axis_radius_phi, axis_z, axis_z_phi;
    periodic_hermite_device(
        phi, axis_R, axis_R_phi, axis_count, static_cast<float>(TWOPI) / nfp,
        axis_radius, axis_radius_phi
    );
    periodic_hermite_device(
        phi, axis_Z, axis_Z_phi, axis_count, static_cast<float>(TWOPI) / nfp,
        axis_z, axis_z_phi
    );
    float sine_phi, cosine_phi, sine_theta, cosine_theta;
    sincosf(phi, &sine_phi, &cosine_phi);
    sincosf(geometric_theta, &sine_theta, &cosine_theta);
    const float R = axis_radius + radius * cosine_theta;
    const float Z = axis_z + radius * sine_theta;
    float s_value, grad_R, grad_Z, grad_phi_coordinate;
    evaluate_psi_device(
        psi_coefficients, mode_a, mode_b, mode_m, mode_kind, psi_coefficient_count,
        nfp, psi_degree, psi_m_tor, psi_a, R, Z, phi,
        axis_radius, axis_z, axis_radius_phi, axis_z_phi,
        s_value, grad_R, grad_Z, grad_phi_coordinate
    );
    float flux = 0.0f;
    float flux_derivative = 0.0f;
    float power = s_value;
    float derivative_power = 1.0f;
    for (int coefficient = 0; coefficient < flux_degree; ++coefficient) {
        flux += flux_coefficients[coefficient] * power;
        flux_derivative += (coefficient + 1) * flux_coefficients[coefficient] * derivative_power;
        power *= s_value;
        derivative_power *= s_value;
    }
    const float physical_rho = sqrtf(fmaxf(flux / edge_flux, 0.0f));
    const float physical_grad_phi = grad_phi_coordinate / R;
    xyz[3 * index] = R * cosine_phi;
    xyz[3 * index + 1] = R * sine_phi;
    xyz[3 * index + 2] = Z;
    grad_s[3 * index] = grad_R * cosine_phi - physical_grad_phi * sine_phi;
    grad_s[3 * index + 1] = grad_R * sine_phi + physical_grad_phi * cosine_phi;
    grad_s[3 * index + 2] = grad_Z;
    const float radius2 = fmaxf(radius * radius, 1.0e-30f);
    const float theta_R = radius * sine_theta / radius2;
    const float theta_Z = -radius * cosine_theta / radius2;
    const float theta_phi_coordinate =
        (radius * cosine_theta * axis_z_phi - radius * sine_theta * axis_radius_phi) / radius2;
    const float physical_theta_phi = theta_phi_coordinate / R;
    grad_theta[3 * index] = theta_R * cosine_phi - physical_theta_phi * sine_phi;
    grad_theta[3 * index + 1] = theta_R * sine_phi + physical_theta_phi * cosine_phi;
    grad_theta[3 * index + 2] = theta_Z;
    grad_phi[3 * index] = -sine_phi / R;
    grad_phi[3 * index + 1] = cosine_phi / R;
    grad_phi[3 * index + 2] = 0.0f;
    s_out[index] = s_value;
    rho_out[index] = physical_rho;
    theta_out[index] = -geometric_theta;
    phi_out[index] = phi;
    R_out[index] = R;
    volume_weight[index] = R;
    flux_derivative_out[index] = flux_derivative;
    const float lower = edge_level * rho_min * rho_min;
    valid[index] = static_cast<unsigned char>(
        isfinite(s_value) && isfinite(physical_rho) && R > 0.0f &&
        s_value >= lower && s_value <= edge_level * (1.0f + 2.0e-5f) &&
        physical_rho <= 1.0001f
    );
    source_indices[index] = index;
}

__global__ void stratify_selected_indices_kernel(
    const int* __restrict__ available_indices,
    int available_count,
    int output_count,
    int* __restrict__ selected_indices
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= output_count) return;
    const int source = min(
        available_count - 1,
        static_cast<int>((static_cast<long long>(index) * available_count) / output_count)
    );
    selected_indices[index] = available_indices[source];
}

__global__ void gather_volume_points_kernel(
    const int* __restrict__ indices,
    int count,
    const float* __restrict__ source_xyz,
    const float* __restrict__ source_grad_s,
    const float* __restrict__ source_grad_theta,
    const float* __restrict__ source_grad_phi,
    const float* __restrict__ source_s,
    const float* __restrict__ source_rho,
    const float* __restrict__ source_theta,
    const float* __restrict__ source_phi,
    const float* __restrict__ source_R,
    const float* __restrict__ source_volume_weight,
    const float* __restrict__ source_flux_derivative,
    float* __restrict__ xyz,
    float* __restrict__ grad_s,
    float* __restrict__ grad_theta,
    float* __restrict__ grad_phi,
    float* __restrict__ s,
    float* __restrict__ rho,
    float* __restrict__ theta,
    float* __restrict__ phi,
    float* __restrict__ R,
    float* __restrict__ volume_weight,
    float* __restrict__ flux_derivative
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const int source = indices[index];
    for (int component = 0; component < 3; ++component) {
        xyz[3 * index + component] = source_xyz[3 * source + component];
        grad_s[3 * index + component] = source_grad_s[3 * source + component];
        grad_theta[3 * index + component] = source_grad_theta[3 * source + component];
        grad_phi[3 * index + component] = source_grad_phi[3 * source + component];
    }
    s[index] = source_s[source];
    rho[index] = source_rho[source];
    theta[index] = source_theta[source];
    phi[index] = source_phi[source];
    R[index] = source_R[source];
    volume_weight[index] = source_volume_weight[source];
    flux_derivative[index] = source_flux_derivative[source];
}

bool build_volume_points_native(
    const DevicePsiData& device_psi,
    const AxisData& axis,
    const PsiData& psi,
    const FluxCalibrationNative& flux,
    int nfp,
    const SgpuScoreConfig& config,
    DeviceVolumePoints& output,
    SgpuScoreResult& result
) {
    const int n_phi = config.volume_phi_count;
    const int n_theta = config.volume_theta_count;
    const int ray_count = n_phi * n_theta;
    const int candidate_target = static_cast<int>(std::ceil(config.volume_point_count * 1.25));
    const int n_radial = (candidate_target + ray_count - 1) / ray_count;
    const int candidate_count = ray_count * n_radial;
    std::vector<float> edge_level{static_cast<float>(flux.edge_level)};
    DeviceBuffer<float> d_edge_level, d_flux_coefficients;
    DeviceBuffer<float> d_ray_coefficients(static_cast<size_t>(ray_count) * (config.psi_poly_degree + 1));
    DeviceBuffer<float> d_boundary_radii(ray_count), d_boundary_residuals(ray_count);
    DeviceVolumePoints candidates;
    DeviceBuffer<unsigned char> d_valid(candidate_count);
    DeviceBuffer<int> d_source_indices(candidate_count), d_available_indices(candidate_count), d_available_count(1);
    if (!copy_to_device(d_edge_level, edge_level) ||
        !copy_to_device(d_flux_coefficients, flux.coefficients) ||
        !d_ray_coefficients.data() || !d_boundary_radii.data() ||
        !d_boundary_residuals.data() || !candidates.allocate(candidate_count) ||
        !d_valid.data() || !d_source_indices.data() || !d_available_indices.data() ||
        !d_available_count.data()) {
        fail_result(&result, "volume point device allocation failed");
        return false;
    }
    constexpr int threads = 256;
    build_ray_coefficients_kernel<<<(ray_count + threads - 1) / threads, threads>>>(
        device_psi.coeffs.data(), device_psi.mode_a.data(), device_psi.mode_b.data(),
        device_psi.mode_m.data(), device_psi.mode_kind.data(), static_cast<int>(psi.coeffs.size()),
        nfp, config.psi_poly_degree, config.psi_m_tor, n_phi, n_theta, 0.417f, 0.613f,
        d_ray_coefficients.data()
    );
    solve_boundary_radii_kernel<<<(ray_count + threads - 1) / threads, threads>>>(
        d_ray_coefficients.data(), ray_count, config.psi_poly_degree,
        static_cast<float>(config.psi_a), d_edge_level.data(), 1,
        config.surface_newton_iters, static_cast<float>(config.surface_newton_tolerance),
        static_cast<float>(config.surface_max_radius_scale),
        d_boundary_radii.data(), d_boundary_residuals.data()
    );
    generate_volume_candidates_kernel<<<(candidate_count + threads - 1) / threads, threads>>>(
        d_boundary_radii.data(), n_phi, n_theta, n_radial, nfp,
        static_cast<float>(config.volume_rho_min), static_cast<float>(flux.edge_level),
        d_flux_coefficients.data(), static_cast<int>(flux.coefficients.size()),
        static_cast<float>(flux.edge_flux), device_psi.coeffs.data(),
        device_psi.mode_a.data(), device_psi.mode_b.data(), device_psi.mode_m.data(),
        device_psi.mode_kind.data(), static_cast<int>(psi.coeffs.size()),
        config.psi_poly_degree, config.psi_m_tor, static_cast<float>(config.psi_a),
        device_psi.axis_R.data(), device_psi.axis_Z.data(),
        device_psi.axis_R_phi.data(), device_psi.axis_Z_phi.data(),
        static_cast<int>(axis.R.size()), candidates.xyz.data(), candidates.grad_s.data(),
        candidates.grad_theta.data(), candidates.grad_phi.data(), candidates.s.data(),
        candidates.rho.data(), candidates.theta.data(), candidates.phi.data(), candidates.R.data(),
        candidates.volume_weight.data(), candidates.flux_derivative.data(), d_valid.data(),
        d_source_indices.data()
    );
    if (!cuda_stage_ok(cudaGetLastError(), result, "volume candidate generation")) return false;
    size_t temporary_bytes = 0;
    cub::DeviceSelect::Flagged(
        nullptr, temporary_bytes, d_source_indices.data(), d_valid.data(),
        d_available_indices.data(), d_available_count.data(), candidate_count
    );
    DeviceBuffer<unsigned char> d_temporary(temporary_bytes);
    if (temporary_bytes > 0 && !d_temporary.data()) {
        fail_result(&result, "volume compaction workspace allocation failed");
        return false;
    }
    cub::DeviceSelect::Flagged(
        d_temporary.data(), temporary_bytes, d_source_indices.data(), d_valid.data(),
        d_available_indices.data(), d_available_count.data(), candidate_count
    );
    int available_count = 0;
    if (!cuda_stage_ok(cudaMemcpy(&available_count, d_available_count.data(), sizeof(int), cudaMemcpyDeviceToHost), result, "volume valid count")) {
        return false;
    }
    const int minimum_count = std::max(
        config.alpha_fit_point_count,
        static_cast<int>(std::ceil(0.6 * config.volume_point_count))
    );
    if (available_count < minimum_count) {
        result.status = SGPU_SCORE_FLUX_REJECTED;
        std::snprintf(result.error_message, sizeof(result.error_message),
                      "volume sampler found %d valid points; minimum is %d", available_count, minimum_count);
        return false;
    }
    const int output_count = std::min(config.volume_point_count, available_count);
    DeviceBuffer<int> d_selected_indices(output_count);
    if (!d_selected_indices.data() || !output.allocate(output_count)) {
        fail_result(&result, "volume output allocation failed");
        return false;
    }
    stratify_selected_indices_kernel<<<(output_count + threads - 1) / threads, threads>>>(
        d_available_indices.data(), available_count, output_count, d_selected_indices.data()
    );
    gather_volume_points_kernel<<<(output_count + threads - 1) / threads, threads>>>(
        d_selected_indices.data(), output_count,
        candidates.xyz.data(), candidates.grad_s.data(), candidates.grad_theta.data(),
        candidates.grad_phi.data(), candidates.s.data(), candidates.rho.data(),
        candidates.theta.data(), candidates.phi.data(), candidates.R.data(),
        candidates.volume_weight.data(), candidates.flux_derivative.data(),
        output.xyz.data(), output.grad_s.data(), output.grad_theta.data(), output.grad_phi.data(),
        output.s.data(), output.rho.data(), output.theta.data(), output.phi.data(), output.R.data(),
        output.volume_weight.data(), output.flux_derivative.data()
    );
    if (!cuda_stage_ok(cudaDeviceSynchronize(), result, "volume point compaction")) return false;
    result.volume_point_count = output_count;
    return true;
}

struct ClebschModesNative {
    std::vector<int> l;
    std::vector<int> m;
    std::vector<int> n;
    std::vector<int> kind;
    std::vector<float> radial_coefficients;
    int radial_order = 0;
};

double factorial_int(int value) {
    double result = 1.0;
    for (int index = 2; index <= value; ++index) result *= index;
    return result;
}

ClebschModesNative build_clebsch_modes_native(
    int radial_order,
    int poloidal_order,
    int toroidal_order
) {
    ClebschModesNative modes;
    modes.radial_order = radial_order;
    const int m_max = std::min(radial_order, poloidal_order);
    for (int m = 0; m <= m_max; ++m) {
        const int n_min = m == 0 ? 1 : -toroidal_order;
        const int n_max = toroidal_order;
        for (int n = n_min; n <= n_max; ++n) {
            for (int l = m; l <= radial_order; l += 2) {
                for (int kind = 0; kind < 2; ++kind) {
                    modes.l.push_back(l);
                    modes.m.push_back(m);
                    modes.n.push_back(n);
                    modes.kind.push_back(kind);
                    const size_t old_size = modes.radial_coefficients.size();
                    modes.radial_coefficients.resize(old_size + radial_order + 1, 0.0f);
                    const int half_plus = (l + m) / 2;
                    const int half_minus = (l - m) / 2;
                    for (int k = 0; k <= half_minus; ++k) {
                        double coefficient = ((k & 1) ? -1.0 : 1.0) * factorial_int(l - k);
                        coefficient /= factorial_int(k) * factorial_int(half_plus - k) *
                                       factorial_int(half_minus - k);
                        modes.radial_coefficients[old_size + l - 2 * k] = static_cast<float>(coefficient);
                    }
                }
            }
        }
    }
    return modes;
}

__global__ void alpha_radial_histogram_kernel(
    const float* __restrict__ rho,
    int point_count,
    int fit_count,
    int radial_bins,
    int* __restrict__ bin_counts
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= fit_count) return;
    const int point = min(
        point_count - 1,
        static_cast<int>((static_cast<long long>(index) * point_count) / fit_count)
    );
    const int bin = min(radial_bins - 1, max(0, static_cast<int>(rho[point] * radial_bins)));
    atomicAdd(bin_counts + bin, 1);
}

__global__ void alpha_prepare_kernel(
    const float* __restrict__ B,
    const float* __restrict__ grad_s,
    const float* __restrict__ grad_theta,
    const float* __restrict__ grad_phi,
    const float* __restrict__ source_rho,
    const float* __restrict__ source_theta,
    const float* __restrict__ source_phi,
    int point_count,
    int fit_count,
    int radial_bins,
    const int* __restrict__ bin_counts,
    float* __restrict__ b_theta,
    float* __restrict__ b_phi,
    float* __restrict__ weights,
    float* __restrict__ rho,
    float* __restrict__ theta,
    float* __restrict__ phi,
    double* __restrict__ diagnostics
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= fit_count) return;
    const int point = min(
        point_count - 1,
        static_cast<int>((static_cast<long long>(index) * point_count) / fit_count)
    );
    float grad_norm2 = 0.0f;
    float field_dot_grad = 0.0f;
    float field_norm2 = 0.0f;
    for (int component = 0; component < 3; ++component) {
        const float field = B[3 * point + component];
        const float gradient = grad_s[3 * point + component];
        grad_norm2 += gradient * gradient;
        field_dot_grad += field * gradient;
        field_norm2 += field * field;
    }
    const float normal_coefficient = field_dot_grad / fmaxf(grad_norm2, 1.0e-30f);
    float along_theta = 0.0f;
    float along_phi = 0.0f;
    float normal_norm2 = 0.0f;
    for (int component = 0; component < 3; ++component) {
        const float normal = normal_coefficient * grad_s[3 * point + component];
        const float tangent = B[3 * point + component] - normal;
        along_theta += tangent * grad_theta[3 * point + component];
        along_phi += tangent * grad_phi[3 * point + component];
        normal_norm2 += normal * normal;
    }
    const float radial = source_rho[point];
    const int bin = min(radial_bins - 1, max(0, static_cast<int>(radial * radial_bins)));
    const float field_norm = sqrtf(fmaxf(field_norm2, 1.0e-30f));
    const float weight = rsqrtf(static_cast<float>(max(bin_counts[bin], 1))) / field_norm;
    b_theta[index] = along_theta;
    b_phi[index] = along_phi;
    weights[index] = weight;
    rho[index] = radial;
    theta[index] = source_theta[point];
    phi[index] = source_phi[point];
    atomic_add_double(diagnostics, static_cast<double>(weight) * weight);
    atomic_add_double(diagnostics + 1, static_cast<double>(normal_norm2));
    atomic_add_double(diagnostics + 2, static_cast<double>(field_norm2));
}

__global__ void normalize_alpha_weights_kernel(
    float* __restrict__ weights,
    int count,
    double weight_norm2,
    const float* __restrict__ b_theta,
    float* __restrict__ rhs,
    int rhs_rows
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const float scale = sqrtf(static_cast<float>(count / fmax(weight_norm2, 1.0e-300)));
        weights[index] *= scale;
        rhs[index] = -weights[index] * b_theta[index];
    }
    if (index >= count && index < rhs_rows) rhs[index] = 0.0f;
}

__device__ inline float evaluate_radial_polynomial(
    const float* coefficients,
    int degree,
    float rho
) {
    float result = coefficients[degree];
    for (int power = degree - 1; power >= 0; --power) result = result * rho + coefficients[power];
    return result;
}

__global__ void assemble_alpha_design_kernel(
    const int* __restrict__ mode_m,
    const int* __restrict__ mode_n,
    const int* __restrict__ mode_kind,
    const float* __restrict__ radial_coefficients,
    int mode_count,
    int radial_order,
    int iota_degree,
    int nfp,
    const float* __restrict__ rho,
    const float* __restrict__ theta,
    const float* __restrict__ phi,
    const float* __restrict__ b_theta,
    const float* __restrict__ b_phi,
    const float* __restrict__ weights,
    int row_count,
    int leading_dimension,
    float* __restrict__ matrix
) {
    const int column = blockIdx.x;
    if (column < mode_count) {
        const int m = mode_m[column];
        const int n = mode_n[column];
        const int kind = mode_kind[column];
        const float* radial_coeff = radial_coefficients + static_cast<size_t>(column) * (radial_order + 1);
        for (int row = threadIdx.x; row < row_count; row += blockDim.x) {
            const float radial = evaluate_radial_polynomial(radial_coeff, radial_order, rho[row]);
            const float argument = m * theta[row] - n * nfp * phi[row];
            float sine, cosine;
            sincosf(argument, &sine, &cosine);
            const float derivative_theta = kind == 0 ? -m * radial * sine : m * radial * cosine;
            const float derivative_phi = kind == 0 ? n * nfp * radial * sine : -n * nfp * radial * cosine;
            matrix[static_cast<size_t>(column) * leading_dimension + row] =
                weights[row] * (derivative_theta * b_theta[row] + derivative_phi * b_phi[row]);
        }
    } else {
        const int power = column - mode_count;
        if (power > iota_degree) return;
        for (int row = threadIdx.x; row < row_count; row += blockDim.x) {
            float radial_power = 1.0f;
            const float u = rho[row] * rho[row];
            for (int index = 0; index < power; ++index) radial_power *= u;
            matrix[static_cast<size_t>(column) * leading_dimension + row] =
                -weights[row] * radial_power * b_phi[row];
        }
    }
}

__global__ void scale_and_augment_alpha_columns_kernel(
    float* __restrict__ matrix,
    int data_rows,
    int leading_dimension,
    int column_count,
    float ridge_sqrt,
    float* __restrict__ scales
) {
    extern __shared__ double shared[];
    const int column = blockIdx.x;
    double sum = 0.0;
    for (int row = threadIdx.x; row < data_rows; row += blockDim.x) {
        const float value = matrix[static_cast<size_t>(column) * leading_dimension + row];
        sum += static_cast<double>(value) * value;
    }
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = fmaxf(static_cast<float>(sqrt(shared[0])), 1.0e-20f);
    if (threadIdx.x == 0) scales[column] = scale;
    __syncthreads();
    for (int row = threadIdx.x; row < data_rows; row += blockDim.x) {
        matrix[static_cast<size_t>(column) * leading_dimension + row] /= scale;
    }
    for (int row = data_rows + threadIdx.x; row < leading_dimension; row += blockDim.x) {
        matrix[static_cast<size_t>(column) * leading_dimension + row] =
            row == data_rows + column ? ridge_sqrt : 0.0f;
    }
}

__global__ void unscale_alpha_solution_kernel(
    const float* __restrict__ scaled_solution,
    const float* __restrict__ scales,
    int count,
    float* __restrict__ solution
) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) solution[index] = scaled_solution[index] / scales[index];
}

struct AlphaFitNative {
    std::vector<float> iota_coefficients;
    double relative_l2 = std::numeric_limits<double>::quiet_NaN();
    double normal_B_relative_l2 = std::numeric_limits<double>::quiet_NaN();
    int column_count = 0;
};

bool fit_alpha_native(
    const DeviceVolumePoints& points,
    const float* d_B,
    int nfp,
    const SgpuScoreConfig& config,
    AlphaFitNative& fit,
    SgpuScoreResult& result
) {
    const auto assembly_started = Clock::now();
    const int fit_count = std::min(config.alpha_fit_point_count, points.count);
    const ClebschModesNative modes = build_clebsch_modes_native(
        config.alpha_radial_order,
        config.alpha_poloidal_order,
        config.alpha_toroidal_order
    );
    const int mode_count = static_cast<int>(modes.m.size());
    const int column_count = mode_count + config.iota_degree + 1;
    const int qr_rows = fit_count + column_count;
    fit.column_count = column_count;
    result.alpha_column_count = column_count;
    DeviceBuffer<int> d_mode_m, d_mode_n, d_mode_kind, d_bin_counts(config.radial_bin_count);
    DeviceBuffer<float> d_radial_coefficients;
    DeviceBuffer<float> d_b_theta(fit_count), d_b_phi(fit_count), d_weights(fit_count);
    DeviceBuffer<float> d_rho(fit_count), d_theta(fit_count), d_phi(fit_count);
    DeviceBuffer<double> d_diagnostics(3);
    DeviceBuffer<float> d_matrix(static_cast<size_t>(qr_rows) * column_count);
    DeviceBuffer<float> d_matrix_reference(static_cast<size_t>(fit_count) * column_count);
    DeviceBuffer<float> d_rhs(qr_rows), d_rhs_reference(fit_count), d_scales(column_count), d_solution(column_count);
    if (!copy_to_device(d_mode_m, modes.m) || !copy_to_device(d_mode_n, modes.n) ||
        !copy_to_device(d_mode_kind, modes.kind) ||
        !copy_to_device(d_radial_coefficients, modes.radial_coefficients) ||
        !d_bin_counts.data() || !d_b_theta.data() || !d_b_phi.data() || !d_weights.data() ||
        !d_rho.data() || !d_theta.data() || !d_phi.data() || !d_diagnostics.data() ||
        !d_matrix.data() || !d_matrix_reference.data() || !d_rhs.data() || !d_rhs_reference.data() ||
        !d_scales.data() || !d_solution.data()) {
        fail_result(&result, "alpha device allocation failed");
        return false;
    }
    cudaMemset(d_bin_counts.data(), 0, d_bin_counts.size() * sizeof(int));
    cudaMemset(d_diagnostics.data(), 0, d_diagnostics.size() * sizeof(double));
    constexpr int threads = 256;
    alpha_radial_histogram_kernel<<<(fit_count + threads - 1) / threads, threads>>>(
        points.rho.data(), points.count, fit_count, config.radial_bin_count, d_bin_counts.data()
    );
    alpha_prepare_kernel<<<(fit_count + threads - 1) / threads, threads>>>(
        d_B, points.grad_s.data(), points.grad_theta.data(), points.grad_phi.data(),
        points.rho.data(), points.theta.data(), points.phi.data(), points.count, fit_count,
        config.radial_bin_count, d_bin_counts.data(), d_b_theta.data(), d_b_phi.data(),
        d_weights.data(), d_rho.data(), d_theta.data(), d_phi.data(), d_diagnostics.data()
    );
    std::array<double, 3> diagnostics{};
    if (!cuda_stage_ok(cudaMemcpy(diagnostics.data(), d_diagnostics.data(), sizeof(diagnostics), cudaMemcpyDeviceToHost), result, "alpha diagnostics") ||
        !cuda_stage_ok(cudaMemset(d_matrix.data(), 0, d_matrix.size() * sizeof(float)), result, "alpha matrix clear")) {
        return false;
    }
    normalize_alpha_weights_kernel<<<(qr_rows + threads - 1) / threads, threads>>>(
        d_weights.data(), fit_count, diagnostics[0], d_b_theta.data(), d_rhs.data(), qr_rows
    );
    assemble_alpha_design_kernel<<<column_count, threads>>>(
        d_mode_m.data(), d_mode_n.data(), d_mode_kind.data(), d_radial_coefficients.data(),
        mode_count, config.alpha_radial_order, config.iota_degree, nfp,
        d_rho.data(), d_theta.data(), d_phi.data(), d_b_theta.data(), d_b_phi.data(),
        d_weights.data(), fit_count, qr_rows, d_matrix.data()
    );
    scale_and_augment_alpha_columns_kernel<<<column_count, threads, threads * sizeof(double)>>>(
        d_matrix.data(), fit_count, qr_rows, column_count,
        static_cast<float>(std::sqrt(std::max(config.alpha_ridge, 0.0))), d_scales.data()
    );
    if (!cuda_stage_ok(cudaMemcpy(d_rhs_reference.data(), d_rhs.data(), fit_count * sizeof(float), cudaMemcpyDeviceToDevice), result, "alpha rhs reference") ||
        !cuda_stage_ok(cudaMemcpy2D(
            d_matrix_reference.data(), static_cast<size_t>(fit_count) * sizeof(float),
            d_matrix.data(), static_cast<size_t>(qr_rows) * sizeof(float),
            static_cast<size_t>(fit_count) * sizeof(float), column_count,
            cudaMemcpyDeviceToDevice
        ), result, "alpha matrix reference") ||
        !cuda_stage_ok(cudaDeviceSynchronize(), result, "alpha assembly")) {
        return false;
    }
    result.timings[SGPU_SCORE_TIME_ALPHA_ASSEMBLE] = seconds_since(assembly_started);
    fit.normal_B_relative_l2 = std::sqrt(diagnostics[1] / std::max(diagnostics[2], 1.0e-300));

    const auto solve_started = Clock::now();
    cublasHandle_t blas = nullptr;
    cusolverDnHandle_t solver = nullptr;
    DeviceBuffer<float> d_tau(column_count);
    DeviceBuffer<int> d_info(1);
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS ||
        cusolverDnCreate(&solver) != CUSOLVER_STATUS_SUCCESS ||
        !d_tau.data() || !d_info.data()) {
        if (blas) cublasDestroy(blas);
        if (solver) cusolverDnDestroy(solver);
        fail_result(&result, "alpha cuBLAS/cuSOLVER initialization failed");
        return false;
    }
    int geqrf_work = 0;
    int ormqr_work = 0;
    cusolverStatus_t solver_status = cusolverDnSgeqrf_bufferSize(
        solver, qr_rows, column_count, d_matrix.data(), qr_rows, &geqrf_work
    );
    if (solver_status == CUSOLVER_STATUS_SUCCESS) {
        solver_status = cusolverDnSormqr_bufferSize(
            solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, qr_rows, 1, column_count,
            d_matrix.data(), qr_rows, d_tau.data(), d_rhs.data(), qr_rows, &ormqr_work
        );
    }
    DeviceBuffer<float> d_work(std::max(geqrf_work, ormqr_work));
    int info = 0;
    if (solver_status != CUSOLVER_STATUS_SUCCESS || !d_work.data()) {
        cublasDestroy(blas);
        cusolverDnDestroy(solver);
        fail_result(&result, "alpha QR workspace query failed");
        return false;
    }
    solver_status = cusolverDnSgeqrf(
        solver, qr_rows, column_count, d_matrix.data(), qr_rows,
        d_tau.data(), d_work.data(), static_cast<int>(d_work.size()), d_info.data()
    );
    cudaMemcpy(&info, d_info.data(), sizeof(int), cudaMemcpyDeviceToHost);
    if (solver_status == CUSOLVER_STATUS_SUCCESS && info == 0) {
        solver_status = cusolverDnSormqr(
            solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, qr_rows, 1, column_count,
            d_matrix.data(), qr_rows, d_tau.data(), d_rhs.data(), qr_rows,
            d_work.data(), static_cast<int>(d_work.size()), d_info.data()
        );
        cudaMemcpy(&info, d_info.data(), sizeof(int), cudaMemcpyDeviceToHost);
    }
    cublasStatus_t blas_status = CUBLAS_STATUS_SUCCESS;
    if (solver_status == CUSOLVER_STATUS_SUCCESS && info == 0) {
        blas_status = cublasStrsv(
            blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
            column_count, d_matrix.data(), qr_rows, d_rhs.data(), 1
        );
    }
    if (solver_status != CUSOLVER_STATUS_SUCCESS || info != 0 || blas_status != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(blas);
        cusolverDnDestroy(solver);
        fail_result(&result, "alpha FP32 QR solve failed");
        return false;
    }
    unscale_alpha_solution_kernel<<<(column_count + threads - 1) / threads, threads>>>(
        d_rhs.data(), d_scales.data(), column_count, d_solution.data()
    );
    DeviceBuffer<float> d_prediction(fit_count);
    if (!d_prediction.data()) {
        cublasDestroy(blas);
        cusolverDnDestroy(solver);
        fail_result(&result, "alpha residual allocation failed");
        return false;
    }
    const float one = 1.0f;
    const float zero = 0.0f;
    const float minus_one = -1.0f;
    blas_status = cublasSgemv(
        blas, CUBLAS_OP_N, fit_count, column_count, &one,
        d_matrix_reference.data(), fit_count, d_rhs.data(), 1, &zero, d_prediction.data(), 1
    );
    if (blas_status == CUBLAS_STATUS_SUCCESS) {
        blas_status = cublasSaxpy(
            blas, fit_count, &minus_one, d_rhs_reference.data(), 1, d_prediction.data(), 1
        );
    }
    float residual_norm = 0.0f;
    float rhs_norm = 0.0f;
    if (blas_status == CUBLAS_STATUS_SUCCESS) {
        blas_status = cublasSnrm2(blas, fit_count, d_prediction.data(), 1, &residual_norm);
    }
    if (blas_status == CUBLAS_STATUS_SUCCESS) {
        blas_status = cublasSnrm2(blas, fit_count, d_rhs_reference.data(), 1, &rhs_norm);
    }
    if (blas_status != CUBLAS_STATUS_SUCCESS ||
        !cuda_stage_ok(cudaDeviceSynchronize(), result, "alpha QR completion")) {
        cublasDestroy(blas);
        cusolverDnDestroy(solver);
        if (result.status != SGPU_SCORE_INTERNAL_ERROR) fail_result(&result, "alpha residual evaluation failed");
        return false;
    }
    fit.relative_l2 = residual_norm / std::max(rhs_norm, 1.0e-20f);
    fit.iota_coefficients.resize(config.iota_degree + 1);
    if (!cuda_stage_ok(cudaMemcpy(
            fit.iota_coefficients.data(), d_solution.data() + mode_count,
            fit.iota_coefficients.size() * sizeof(float), cudaMemcpyDeviceToHost
        ), result, "alpha iota copy")) {
        cublasDestroy(blas);
        cusolverDnDestroy(solver);
        return false;
    }
    cublasDestroy(blas);
    cusolverDnDestroy(solver);
    result.timings[SGPU_SCORE_TIME_ALPHA_QR] = seconds_since(solve_started);
    return true;
}

__global__ void compute_qs_metric_kernel(
    const float* __restrict__ B,
    const float* __restrict__ grad_B,
    const float* __restrict__ grad_s,
    const float* __restrict__ flux_derivative,
    const float* __restrict__ rho,
    const float* __restrict__ volume_weight,
    int point_count,
    const float* __restrict__ iota_coefficients,
    int iota_degree,
    int helicity_M,
    int helicity_N,
    double G,
    float edge_rho_threshold,
    float* __restrict__ absolute_normalized,
    double* __restrict__ sums
) {
    const int point = blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= point_count) return;
    float magnitude2 = 0.0f;
    for (int component = 0; component < 3; ++component) {
        const float value = B[3 * point + component];
        magnitude2 += value * value;
    }
    const float magnitude = sqrtf(fmaxf(magnitude2, 1.0e-30f));
    float grad_magnitude[3];
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        float sum = 0.0f;
        for (int component = 0; component < 3; ++component) {
            sum += grad_B[9 * point + 3 * component + coordinate] * B[3 * point + component];
        }
        grad_magnitude[coordinate] = sum / magnitude;
    }
    float grad_psi[3];
    for (int component = 0; component < 3; ++component) {
        grad_psi[component] = flux_derivative[point] * grad_s[3 * point + component];
    }
    const float cross_x = B[3 * point + 1] * grad_psi[2] - B[3 * point + 2] * grad_psi[1];
    const float cross_y = B[3 * point + 2] * grad_psi[0] - B[3 * point] * grad_psi[2];
    const float cross_z = B[3 * point] * grad_psi[1] - B[3 * point + 1] * grad_psi[0];
    const float A = cross_x * grad_magnitude[0] + cross_y * grad_magnitude[1] +
                    cross_z * grad_magnitude[2];
    const float C = B[3 * point] * grad_magnitude[0] + B[3 * point + 1] * grad_magnitude[1] +
                    B[3 * point + 2] * grad_magnitude[2];
    const float u = rho[point] * rho[point];
    float iota = 0.0f;
    float power = 1.0f;
    for (int degree = 0; degree <= iota_degree; ++degree) {
        iota += iota_coefficients[degree] * power;
        power *= u;
    }
    const double f_c = (helicity_M * static_cast<double>(iota) - helicity_N) * A -
                       helicity_M * G * C;
    const double normalized = f_c /
        fmax(static_cast<double>(magnitude) * magnitude * magnitude, 1.0e-30);
    const double weight = volume_weight[point];
    absolute_normalized[point] = static_cast<float>(fabs(normalized));
    atomic_add_double(sums, weight);
    atomic_add_double(sums + 1, weight * normalized * normalized);
    if (rho[point] >= edge_rho_threshold) {
        atomic_add_double(sums + 2, weight);
        atomic_add_double(sums + 3, weight * normalized * normalized);
    }
}

bool compute_qs_metric_native(
    const DeviceVolumePoints& points,
    const float* d_B,
    const float* d_grad_B,
    const AlphaFitNative& alpha,
    const double* currents_a,
    int n_base_coils,
    int nfp,
    const SgpuScoreConfig& config,
    SgpuScoreResult& result
) {
    DeviceBuffer<float> d_iota;
    DeviceBuffer<float> d_absolute(points.count);
    DeviceBuffer<double> d_sums(4);
    if (!copy_to_device(d_iota, alpha.iota_coefficients) || !d_absolute.data() || !d_sums.data() ||
        !cuda_stage_ok(cudaMemset(d_sums.data(), 0, d_sums.size() * sizeof(double)), result, "QS reduction clear")) {
        if (result.status != SGPU_SCORE_INTERNAL_ERROR) fail_result(&result, "QS allocation failed");
        return false;
    }
    double current_sum = 0.0;
    for (int index = 0; index < n_base_coils; ++index) current_sum += std::abs(currents_a[index]);
    const double G = 4.0e-7 * PI * 2.0 * nfp * current_sum;
    const float edge_threshold = static_cast<float>(
        config.volume_rho_min + (1.0 - config.volume_rho_min) *
        (config.radial_bin_count - 1.0) / config.radial_bin_count
    );
    constexpr int threads = 256;
    compute_qs_metric_kernel<<<(points.count + threads - 1) / threads, threads>>>(
        d_B, d_grad_B, points.grad_s.data(), points.flux_derivative.data(),
        points.rho.data(), points.volume_weight.data(), points.count,
        d_iota.data(), config.iota_degree, config.target_M, config.target_N,
        G, edge_threshold, d_absolute.data(), d_sums.data()
    );
    if (!cuda_stage_ok(cudaDeviceSynchronize(), result, "QS metric kernel")) return false;
    thrust::device_ptr<float> begin(d_absolute.data());
    thrust::sort(begin, begin + points.count);
    const int percentile_index = std::min(points.count - 1, static_cast<int>(std::floor(0.95 * (points.count - 1))));
    float p95 = 0.0f;
    std::array<double, 4> sums{};
    if (!cuda_stage_ok(cudaMemcpy(&p95, d_absolute.data() + percentile_index, sizeof(float), cudaMemcpyDeviceToHost), result, "QS p95 copy") ||
        !cuda_stage_ok(cudaMemcpy(sums.data(), d_sums.data(), sizeof(sums), cudaMemcpyDeviceToHost), result, "QS sums copy")) {
        return false;
    }
    result.qs_global_error = std::sqrt(sums[1] / std::max(sums[0], 1.0e-300));
    result.qs_edge_error = std::sqrt(sums[3] / std::max(sums[2], 1.0e-300));
    result.qs_abs_p95 = p95;
    return true;
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
    DevicePsiData device_psi;
    if (!upload_psi_data(psi, axis, device_psi)) {
        fail_result(&result, "psi/axis upload for downstream pipeline failed");
        return false;
    }
    auto started = Clock::now();
    FluxCalibrationNative flux;
    const bool flux_ok = calibrate_flux_native(
        field, device_psi, axis, psi, nfp, surface.level, config, flux, result
    );
    result.timings[SGPU_SCORE_TIME_FLUX] = seconds_since(started);
    if (result.status == SGPU_SCORE_INTERNAL_ERROR && result.error_message[0]) return false;
    if (!flux_ok) {
        result.status = SGPU_SCORE_FLUX_REJECTED;
        result.stage_completed = SCORE_STAGE_FLUX;
        return false;
    }
    result.flux_edge = flux.edge_flux;
    result.flux_fit_relative_rms = flux.fit_relative_rms;
    result.flux_section_relative_std_edge = flux.section_relative_std_edge;
    result.flux_boundary_residual_max = flux.boundary_residual_max;
    result.flux_derivative_min = flux.derivative_min;
    result.flux_derivative_max = flux.derivative_max;
    result.surface_volume = flux.volume;
    result.surface_effective_minor_radius = flux.effective_minor_radius;
    result.surface_inverse_aspect_ratio = flux.effective_minor_radius /
        std::max(flux.major_radius, 1.0e-30);
    result.stage_completed = SCORE_STAGE_FLUX;

    DeviceVolumePoints points;
    started = Clock::now();
    if (!build_volume_points_native(device_psi, axis, psi, flux, nfp, config, points, result)) {
        return false;
    }
    result.timings[SGPU_SCORE_TIME_VOLUME_POINTS] = seconds_since(started);
    DeviceBuffer<float> d_B(static_cast<size_t>(points.count) * 3);
    DeviceBuffer<float> d_grad_B(static_cast<size_t>(points.count) * 9);
    if (!d_B.data() || !d_grad_B.data()) {
        fail_result(&result, "volume field allocation failed");
        return false;
    }
    started = Clock::now();
    if (sgpu_internal_eval_B_grad_f32_device(
            field, points.xyz.data(), d_B.data(), d_grad_B.data(), points.count) ||
        !cuda_stage_ok(cudaDeviceSynchronize(), result, "volume B/grad(B) evaluation")) {
        if (result.status != SGPU_SCORE_INTERNAL_ERROR || !result.error_message[0]) {
            fail_from_backend(&result, "volume B/grad(B) evaluation");
        }
        return false;
    }
    result.timings[SGPU_SCORE_TIME_FIELD_VOLUME] = seconds_since(started);
    AlphaFitNative alpha;
    if (!fit_alpha_native(points, d_B.data(), nfp, config, alpha, result)) {
        if (result.status != SGPU_SCORE_INTERNAL_ERROR) result.status = SGPU_SCORE_ALPHA_FAILED;
        return false;
    }
    result.alpha_relative_l2 = alpha.relative_l2;
    result.alpha_normal_B_relative_l2 = alpha.normal_B_relative_l2;
    result.iota_min = *std::min_element(alpha.iota_coefficients.begin(), alpha.iota_coefficients.end());
    result.iota_max = *std::max_element(alpha.iota_coefficients.begin(), alpha.iota_coefficients.end());
    if (config.iota_degree == 0) result.iota_max = result.iota_min;
    result.stage_completed = SCORE_STAGE_ALPHA;
    started = Clock::now();
    if (!compute_qs_metric_native(
            points, d_B.data(), d_grad_B.data(), alpha, currents_a,
            n_base_coils, nfp, config, result)) {
        return false;
    }
    result.timings[SGPU_SCORE_TIME_QS_METRICS] = seconds_since(started);
    const double section_score = q_down(
        result.flux_section_relative_std_edge, config.score_flux_section_std_scale, 1.0
    );
    const double boundary_score = q_down(
        result.flux_boundary_residual_max, config.score_flux_boundary_residual_scale, 1.0
    );
    const double flux_score = blend({{0.75, section_score}, {0.25, boundary_score}});
    const double normal_score = q_down(
        result.alpha_normal_B_relative_l2, config.score_alpha_normal_B_scale, 1.0
    );
    const double alpha_score = q_down(
        result.alpha_relative_l2, config.score_alpha_relative_l2_scale, 1.0
    );
    result.components[SGPU_SCORE_COMPONENT_COORDINATE] = blend({
        {0.35, flux_score}, {0.35, normal_score}, {0.20, alpha_score}, {0.10, 1.0},
    });
    const double global_score = q_down(result.qs_global_error, config.score_qs_global_scale, 0.9);
    const double edge_score = q_down(result.qs_edge_error, config.score_qs_edge_scale, 0.9, global_score);
    const double residual_score = blend({{0.80, global_score}, {0.20, edge_score}});
    const double size_score = q_up(
        result.surface_inverse_aspect_ratio, config.score_surface_inverse_aspect_scale, 2.0
    );
    result.components[SGPU_SCORE_COMPONENT_VOLUME_QS] = residual_score * (0.35 + 0.65 * size_score);
    result.status = SGPU_SCORE_OK;
    result.stage_completed = SCORE_STAGE_QS;
    return true;
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
        const bool downstream_ok = run_downstream_gpu(
                field, currents_a, n_base_coils, nfp, axis, psi,
                *best_strict, *config, *result);
        fill_early_components(*config, coil, axis, psi, screens, *result);
        if (!downstream_ok) {
            if (result->status == SGPU_SCORE_INTERNAL_ERROR) return_code = 1;
            break;
        }
    } while (false);

    sgpu_destroy_field(field);
    finalize_score(*config, *result);
    result->timings[SGPU_SCORE_TIME_TOTAL] = seconds_since(total_started);
    return return_code;
}

}  // extern "C"
