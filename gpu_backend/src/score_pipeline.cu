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

double q_saturating_up(double value, double saturation, double fallback = 0.0) {
    if (!std::isfinite(value) || value <= 0.0 || !(saturation > 0.0)) return fallback;
    const double x = clip01(value / saturation);
    return x * x * (3.0 - 2.0 * x);
}

double minimum_absolute_iota(double iota_min, double iota_max) {
    if (!std::isfinite(iota_min) || !std::isfinite(iota_max)) return 0.0;
    if (iota_min <= 0.0 && iota_max >= 0.0) return 0.0;
    return std::min(std::abs(iota_min), std::abs(iota_max));
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
    const double nan = std::numeric_limits<double>::quiet_NaN();
    result->score = nan;
    result->axis_R = nan;
    result->axis_Z = nan;
    result->axis_residual = nan;
    result->axis_topology_trace = nan;
    result->axis_topology_det = nan;
    result->axis_ellipse_aspect = nan;
    result->psi_train_rms = nan;
    result->psi_angle_mean = nan;
    result->psi_angle_p95 = nan;
    result->psi_angle_l2 = nan;
    result->surface_level = nan;
    result->surface_drift_relative_p95 = nan;
    result->surface_one_period_drift_relative_p95 = nan;
    result->surface_effective_minor_radius = nan;
    result->surface_inverse_aspect_ratio = nan;
    result->surface_volume = nan;
    result->flux_edge = nan;
    result->flux_fit_relative_rms = nan;
    result->flux_section_relative_std_edge = nan;
    result->flux_boundary_residual_max = nan;
    result->flux_derivative_min = nan;
    result->flux_derivative_max = nan;
    result->alpha_relative_l2 = nan;
    result->alpha_normal_B_relative_l2 = nan;
    result->iota_min = nan;
    result->iota_max = nan;
    result->score_surface_size = nan;
    result->score_iota = nan;
    result->score_qs_residual = nan;
    result->score_volume_qs_size_factor = nan;
    result->score_volume_qs_iota_factor = nan;
    result->score_before_qh_iota_gate = nan;
    result->score_qh_total_iota_factor = nan;
    result->score_qh_helicity_advantage = 0.0;
    result->score_qh_helicity_quality = 0.0;
    result->score_qh_total_helicity_factor = nan;
    result->qs_global_error = nan;
    result->qs_edge_error = nan;
    result->qs_qa_global_error = nan;
    result->qs_qp_global_error = nan;
    result->qs_vacuum_G = nan;
    result->qs_target_global_error_per_helicity = nan;
    result->qs_target_edge_error_per_helicity = nan;
    result->qs_qa_global_error_per_helicity = nan;
    result->qs_qp_global_error_raw = nan;
    result->qs_qp_global_error_per_helicity = nan;
    result->qs_abs_p95 = nan;
    result->qs_abs_p95_per_helicity = nan;
    result->volume_valid_fraction = nan;
    result->volume_weight_effective_fraction = nan;
    result->edge_weight_effective_fraction = nan;
    result->surface_confidence_mean = nan;
    result->surface_confidence_edge = nan;
    result->surface_effective_level = nan;
    result->surface_confidence_risk = nan;
    result->axis_hint_distance = nan;
    result->coil_length_mean = nan;
    result->coil_curvature_p95 = nan;
    result->coil_curvature_max = nan;
    result->coil_min_intercoil_distance = nan;
    result->coil_min_axis_distance = nan;
    result->coil_high_mode_energy_fraction = nan;
    result->coil_current_abs_max_a = nan;
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

double axis_topology_stability_margin(const AxisCandidate& candidate) {
    if (!(candidate.topology_det > 0.0) ||
        !std::isfinite(candidate.topology_trace) ||
        !std::isfinite(candidate.topology_det)) {
        return -std::numeric_limits<double>::infinity();
    }
    return 2.0 - std::abs(candidate.topology_trace) / std::sqrt(candidate.topology_det);
}

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

void assign_axis_topology(
    AxisCandidate& candidate,
    double r_plus,
    double r_minus,
    double z_plus,
    double z_minus,
    double r_end_r_plus,
    double r_end_r_minus,
    double z_end_r_plus,
    double z_end_r_minus,
    double r_end_z_plus,
    double r_end_z_minus,
    double z_end_z_plus,
    double z_end_z_minus
) {
    const double denom_R = std::max(r_plus - r_minus, 1.0e-300);
    const double denom_Z = std::max(z_plus - z_minus, 1.0e-300);
    const double a = (r_end_r_plus - r_end_r_minus) / denom_R;
    const double c = (z_end_r_plus - z_end_r_minus) / denom_R;
    const double b = (r_end_z_plus - r_end_z_minus) / denom_Z;
    const double d = (z_end_z_plus - z_end_z_minus) / denom_Z;
    const double trace = a + d;
    const double determinant = a * d - b * c;
    candidate.topology_trace = trace;
    candidate.topology_det = determinant;
    candidate.elliptic = determinant > 0.0 &&
        std::abs(trace / std::sqrt(determinant)) < 2.0;
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
            candidate.ellipse_aspect = std::sqrt(eig1 / eig0);
        }
    }
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
        assign_axis_topology(
            candidates[j],
            R[j], R[count + j], Z[2 * count + j], Z[3 * count + j],
            R_end[j], R_end[count + j], Z_end[j], Z_end[count + j],
            R_end[2 * count + j], R_end[3 * count + j],
            Z_end[2 * count + j], Z_end[3 * count + j]
        );
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
    std::vector<AxisCandidate>& candidates,
    bool fallback,
    double* timings
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
    auto substage_started = Clock::now();
    if (!trace_map(field, R, Z, nfp, config.axis_trace_steps, false, R_end, Z_end)) return false;
    timings[fallback ? SGPU_SCORE_TIME_AXIS_FALLBACK_GRID_TRACE
                     : SGPU_SCORE_TIME_AXIS_PRIMARY_GRID_TRACE] += seconds_since(substage_started);
    std::vector<double> dR(R.size()), dZ(Z.size());
    for (size_t i = 0; i < R.size(); ++i) {
        dR[i] = R_end[i] - R[i];
        dZ[i] = Z_end[i] - Z[i];
    }
    substage_started = Clock::now();
    candidates = find_grid_candidates(rs, zs, dR, dZ, max_candidates);
    timings[SGPU_SCORE_TIME_AXIS_CANDIDATE_EXTRACT] += seconds_since(substage_started);
    substage_started = Clock::now();
    if (!refine_axis_candidates(field, candidates, domain, nfp, config, newton_iters)) return false;
    timings[SGPU_SCORE_TIME_AXIS_CANDIDATE_REFINE] += seconds_since(substage_started);
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
        substage_started = Clock::now();
        if (!trace_map(field, verify_R, verify_Z, nfp, config.axis_trace_steps, true, end_R, end_Z)) return false;
        timings[SGPU_SCORE_TIME_AXIS_FP64_VERIFY] += seconds_since(substage_started);
        for (size_t i = 0; i < candidates.size(); ++i) {
            candidates[i].residual = std::hypot(end_R[i] - verify_R[i], end_Z[i] - verify_Z[i]);
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.residual < rhs.residual;
        });
        substage_started = Clock::now();
        classify_axis_topology(field, candidates, domain, nfp, config);
        timings[SGPU_SCORE_TIME_AXIS_TOPOLOGY] += seconds_since(substage_started);
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
    bool used_hint = false;
    bool branch_lost = false;
    double hint_distance = std::numeric_limits<double>::quiet_NaN();
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
    AxisData& axis,
    double& trace_time,
    double* timings
) {
    trace_time = 0.0;
    auto substage_started = Clock::now();
    const AxisDomain domain = build_axis_domain(
        coeffs_x, coeffs_y, coeffs_z, n_base_coils, n_coeff, config
    );
    timings[SGPU_SCORE_TIME_AXIS_DOMAIN] += seconds_since(substage_started);
    std::vector<AxisCandidate> candidates;
    if (config.axis_hint_enabled) {
        const bool hint_in_domain = config.axis_hint_R >= domain.r_min &&
            config.axis_hint_R <= domain.r_max &&
            config.axis_hint_Z >= domain.z_min && config.axis_hint_Z <= domain.z_max;
        if (!hint_in_domain) {
            axis.hint_distance = std::numeric_limits<double>::infinity();
            if (config.axis_hint_require_continuation) {
                axis.branch_lost = true;
                return true;
            }
        } else {
            candidates.push_back({config.axis_hint_R, config.axis_hint_Z});
        }
    }
    if (config.axis_hint_enabled && !candidates.empty()) {
        substage_started = Clock::now();
        if (!refine_axis_candidates(
                field, candidates, domain, nfp, config, config.axis_newton_iters)) {
            return false;
        }
        timings[SGPU_SCORE_TIME_AXIS_CANDIDATE_REFINE] += seconds_since(substage_started);
        if (!candidates.empty()) {
            const double span = std::max(
                domain.r_max - domain.r_min, domain.z_max - domain.z_min
            );
            const double h = std::max(config.axis_fd_absolute, config.axis_fd_relative * span);
            std::vector<double> verify_R{
                candidates[0].R,
                std::min(domain.r_max, candidates[0].R + h),
                std::max(domain.r_min, candidates[0].R - h),
                candidates[0].R,
                candidates[0].R,
            };
            std::vector<double> verify_Z{
                candidates[0].Z,
                candidates[0].Z,
                candidates[0].Z,
                std::min(domain.z_max, candidates[0].Z + h),
                std::max(domain.z_min, candidates[0].Z - h),
            };
            std::vector<double> end_R, end_Z;
            substage_started = Clock::now();
            if (!trace_map(
                    field, verify_R, verify_Z, nfp, config.axis_trace_steps,
                    true, end_R, end_Z)) {
                return false;
            }
            timings[SGPU_SCORE_TIME_AXIS_FP64_VERIFY] += seconds_since(substage_started);
            candidates[0].residual = std::hypot(
                end_R[0] - verify_R[0], end_Z[0] - verify_Z[0]
            );
            substage_started = Clock::now();
            assign_axis_topology(
                candidates[0],
                verify_R[1], verify_R[2], verify_Z[3], verify_Z[4],
                end_R[1], end_R[2], end_Z[1], end_Z[2],
                end_R[3], end_R[4], end_Z[3], end_Z[4]
            );
            timings[SGPU_SCORE_TIME_AXIS_TOPOLOGY] += seconds_since(substage_started);
            axis.hint_distance = std::hypot(
                candidates[0].R - config.axis_hint_R,
                candidates[0].Z - config.axis_hint_Z
            );
            axis.used_hint = candidates[0].residual <= config.axis_tolerance &&
                candidates[0].elliptic &&
                axis.hint_distance <= config.axis_hint_max_distance;
        }
        if (!axis.used_hint) {
            candidates.clear();
            if (config.axis_hint_require_continuation) {
                axis.branch_lost = true;
                return true;
            }
        }
    }
    if (!axis.used_hint) {
        if (!search_axis_grid(
                field, domain, nfp, config, config.axis_grid,
                config.axis_max_candidates, config.axis_newton_iters, candidates,
                false, timings)) {
            return false;
        }
    }
    auto eligible = [](const AxisCandidate& candidate, double tolerance) {
        return candidate.residual <= tolerance && candidate.elliptic;
    };
    auto robust = [&](const AxisCandidate& candidate) {
        return eligible(candidate, config.axis_tolerance) &&
            axis_topology_stability_margin(candidate) >= config.axis_topology_margin;
    };
    bool has_robust = std::any_of(candidates.begin(), candidates.end(), [&](const auto& candidate) {
        return robust(candidate);
    });
    if (!axis.used_hint && !has_robust && nfp <= config.axis_fallback_max_nfp) {
        std::vector<AxisCandidate> fallback;
        if (!search_axis_grid(
                field, domain, nfp, config, config.axis_fallback_grid,
                config.axis_fallback_max_candidates, config.axis_fallback_newton_iters,
                fallback, true, timings)) {
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
    axis.selected = *std::min_element(elliptic.begin(), elliptic.end(), [&](const auto& lhs, const auto& rhs) {
        const bool lhs_robust = robust(lhs);
        const bool rhs_robust = robust(rhs);
        if (lhs_robust != rhs_robust) return lhs_robust > rhs_robust;
        if (lhs.ellipse_aspect != rhs.ellipse_aspect) return lhs.ellipse_aspect < rhs.ellipse_aspect;
        return lhs.residual < rhs.residual;
    });
    axis.R.resize(config.axis_sample_count);
    axis.Z.resize(config.axis_sample_count);
    axis.R_phi.resize(config.axis_sample_count);
    axis.Z_phi.resize(config.axis_sample_count);
    const auto trace_started = Clock::now();
    const int trace_code = sgpu_trace_axis_samples(
            field, axis.selected.R, axis.selected.Z, nfp,
            config.axis_trace_steps, config.axis_sample_count,
            axis.R.data(), axis.Z.data(), axis.R_phi.data(), axis.Z_phi.data());
    trace_time = seconds_since(trace_started);
    if (trace_code) {
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
        config.psi_solver_mode, config.psi_precision_mode,
        psi.coeffs.data(), &psi.train_rms, stats.data(), static_cast<int>(stats.size())
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
    double one_period_relative_drift_p95 = std::numeric_limits<double>::infinity();
    int long_trace_periods_completed = 0;
    bool stable = false;
    bool strict = false;
    bool verified = false;
    bool long_verified = false;
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
    std::vector<SurfaceScreen>& screens,
    double* timings
) {
    const int levels = config.surface_level_count;
    const int theta_count = config.surface_theta_count;
    const size_t count = static_cast<size_t>(levels) * theta_count;
    std::vector<double> R(count), Z(count), radii(count), theta(count), R_end, Z_end;
    auto substage_started = Clock::now();
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
    timings[SGPU_SCORE_TIME_SURFACE_RAY_ROOTS] += seconds_since(substage_started);
    substage_started = Clock::now();
    if (!trace_map(field, R, Z, nfp, config.surface_trace_steps, false, R_end, Z_end)) return false;
    timings[SGPU_SCORE_TIME_SURFACE_MIXED_TRACE] += seconds_since(substage_started);
    substage_started = Clock::now();
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
        screen.one_period_relative_drift_p95 = screen.relative_drift_p95;
        screen.stable = screen.drift_p95 <= config.surface_drift_absolute_tolerance &&
                        screen.relative_drift_p95 <= 0.30 &&
                        screen.radius_max < config.surface_max_radius_scale * config.psi_a * 0.999;
        screen.strict = screen.stable &&
                        screen.relative_drift_p95 <= config.surface_drift_relative_tolerance;
    }
    timings[SGPU_SCORE_TIME_SURFACE_MIXED_REDUCE] += seconds_since(substage_started);

    std::vector<int> verify_indices;
    // Keep the expensive long-horizon path bounded: only the six largest
    // one-period candidates participate in high-precision verification.
    for (int index = levels - 1; index >= 0 && verify_indices.size() < 6; --index) {
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
        substage_started = Clock::now();
        if (!trace_map(field, verify_R, verify_Z, nfp, config.surface_trace_steps, true, verify_R_end, verify_Z_end)) return false;
        timings[SGPU_SCORE_TIME_SURFACE_FP64_TRACE] += seconds_since(substage_started);
        substage_started = Clock::now();
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
            screens[level_index].one_period_relative_drift_p95 =
                screens[level_index].relative_drift_p95;
            screens[level_index].stable =
                screens[level_index].drift_p95 <= config.surface_drift_absolute_tolerance &&
                screens[level_index].relative_drift_p95 <= 0.30 &&
                screens[level_index].radius_max < config.surface_max_radius_scale * config.psi_a * 0.999;
            screens[level_index].strict = screens[level_index].stable &&
                screens[level_index].relative_drift_p95 <= config.surface_drift_relative_tolerance;
            screens[level_index].verified = true;
        }
        timings[SGPU_SCORE_TIME_SURFACE_FP64_REDUCE] += seconds_since(substage_started);
    }
    return true;
}

double smooth_tail_risk(const std::vector<double>& values, double temperature) {
    if (values.empty()) return std::numeric_limits<double>::infinity();
    const double maximum = *std::max_element(values.begin(), values.end());
    if (!std::isfinite(maximum)) return std::numeric_limits<double>::infinity();
    double sum = 0.0;
    for (double value : values) {
        sum += std::exp((value - maximum) / temperature);
    }
    return maximum + temperature * std::log(sum / values.size());
}

double logistic_confidence(double risk, const SgpuScoreConfig& config) {
    if (!std::isfinite(risk)) return 0.0;
    const double argument = (risk - config.surface_confidence_drift_center) /
        config.surface_confidence_drift_temperature;
    if (argument >= 40.0) return 0.0;
    if (argument <= -40.0) return 1.0;
    return 1.0 / (1.0 + std::exp(argument));
}

void project_nonincreasing_confidence(
    std::vector<double>& confidence,
    const SgpuScoreConfig& config
) {
    struct Block {
        int begin;
        int end;
        double weight;
        double mean;
    };
    std::vector<Block> blocks;
    blocks.reserve(confidence.size());
    double previous_level = 0.0;
    for (int index = 0; index < static_cast<int>(confidence.size()); ++index) {
        const double level = config.surface_levels[index];
        const double weight = std::max(level - previous_level, 1.0e-12);
        previous_level = level;
        blocks.push_back({index, index + 1, weight, confidence[index]});
        while (blocks.size() >= 2 &&
               blocks[blocks.size() - 2].mean < blocks.back().mean) {
            const Block right = blocks.back();
            blocks.pop_back();
            Block& left = blocks.back();
            const double total_weight = left.weight + right.weight;
            left.mean = (left.weight * left.mean + right.weight * right.mean) /
                total_weight;
            left.weight = total_weight;
            left.end = right.end;
        }
    }
    for (const Block& block : blocks) {
        for (int index = block.begin; index < block.end; ++index) {
            confidence[index] = block.mean;
        }
    }
}

bool screen_surface_confidence_native(
    void* field,
    const AxisData& axis,
    const PsiData& psi,
    int nfp,
    const SgpuScoreConfig& config,
    std::vector<SurfaceScreen>& screens,
    SgpuScoreResult& result
) {
    const int levels = config.surface_level_count;
    const int theta_count = config.surface_theta_count;
    const size_t count = static_cast<size_t>(levels) * theta_count;
    std::vector<double> R(count), Z(count), radii(count), radius_mean(levels, 0.0);
    std::vector<double> radius_max(levels, 0.0), R_end, Z_end;
    auto substage_started = Clock::now();
    for (int level_index = 0; level_index < levels; ++level_index) {
        for (int theta_index = 0; theta_index < theta_count; ++theta_index) {
            const size_t index = static_cast<size_t>(level_index) * theta_count + theta_index;
            const double theta = TWOPI * theta_index / theta_count;
            std::array<double, 25> polynomial;
            ray_polynomial_phi0(psi, config, theta, polynomial);
            radii[index] = solve_ray_radius(
                polynomial, config.surface_levels[level_index], config
            );
            radius_mean[level_index] += radii[index] / theta_count;
            radius_max[level_index] = std::max(radius_max[level_index], radii[index]);
            R[index] = axis.R[0] + radii[index] * std::cos(theta);
            Z[index] = axis.Z[0] + radii[index] * std::sin(theta);
        }
    }
    result.timings[SGPU_SCORE_TIME_SURFACE_RAY_ROOTS] += seconds_since(substage_started);

    std::vector<double> maximum_relative_drift(count, 0.0);
    for (int period = 0; period < config.surface_confidence_periods; ++period) {
        substage_started = Clock::now();
        if (!trace_map(field, R, Z, nfp, config.surface_trace_steps, false, R_end, Z_end)) {
            return false;
        }
        result.timings[SGPU_SCORE_TIME_SURFACE_MIXED_TRACE] += seconds_since(substage_started);
        substage_started = Clock::now();
        for (int level_index = 0; level_index < levels; ++level_index) {
            for (int theta_index = 0; theta_index < theta_count; ++theta_index) {
                const size_t index = static_cast<size_t>(level_index) * theta_count + theta_index;
                double value, gR, gZ, gPhi;
                evaluate_psi_host(
                    psi, axis, config, nfp, R_end[index], Z_end[index], TWOPI / nfp,
                    value, gR, gZ, gPhi
                );
                const double gradient_norm = std::sqrt(
                    gR * gR + gZ * gZ + std::pow(gPhi / R_end[index], 2.0)
                );
                const double distance = std::abs(value - config.surface_levels[level_index]) /
                    std::max(gradient_norm, 1.0e-14);
                const double relative = distance / std::max(radius_mean[level_index], 1.0e-14);
                maximum_relative_drift[index] = std::max(
                    maximum_relative_drift[index],
                    std::isfinite(relative) ? relative : std::numeric_limits<double>::infinity()
                );
            }
        }
        R.swap(R_end);
        Z.swap(Z_end);
        result.timings[SGPU_SCORE_TIME_SURFACE_MIXED_REDUCE] += seconds_since(substage_started);
    }

    substage_started = Clock::now();
    std::vector<double> risk(levels), confidence(levels);
    for (int level_index = 0; level_index < levels; ++level_index) {
        const size_t offset = static_cast<size_t>(level_index) * theta_count;
        std::vector<double> level_values(
            maximum_relative_drift.begin() + offset,
            maximum_relative_drift.begin() + offset + theta_count
        );
        risk[level_index] = smooth_tail_risk(
            level_values, config.surface_confidence_smoothmax_temperature
        );
        const double radius_limit = config.surface_max_radius_scale * config.psi_a;
        const double radius_margin = (0.995 * radius_limit - radius_max[level_index]) /
            std::max(0.003 * radius_limit, 1.0e-14);
        const double radius_confidence = radius_margin >= 40.0
            ? 1.0
            : radius_margin <= -40.0
                ? 0.0
                : 1.0 / (1.0 + std::exp(-radius_margin));
        confidence[level_index] = logistic_confidence(risk[level_index], config) *
            radius_confidence;
    }
    project_nonincreasing_confidence(confidence, config);

    double effective_level = 0.0;
    double previous_level = 0.0;
    double previous_confidence = 1.0;
    for (int level_index = 0; level_index < levels; ++level_index) {
        const double level = config.surface_levels[level_index];
        effective_level += 0.5 * (previous_confidence + confidence[level_index]) *
            (level - previous_level);
        previous_level = level;
        previous_confidence = confidence[level_index];
    }
    effective_level = std::min(
        config.surface_levels[levels - 1], std::max(0.0, effective_level)
    );
    const auto interpolate = [&](const std::vector<double>& values, double initial) {
        if (effective_level <= config.surface_levels[0]) {
            const double weight = effective_level / config.surface_levels[0];
            return initial + weight * (values[0] - initial);
        }
        for (int index = 1; index < levels; ++index) {
            if (effective_level <= config.surface_levels[index]) {
                const double weight = (effective_level - config.surface_levels[index - 1]) /
                    (config.surface_levels[index] - config.surface_levels[index - 1]);
                return values[index - 1] + weight * (values[index] - values[index - 1]);
            }
        }
        return values.back();
    };
    const double effective_risk = interpolate(risk, 0.0);
    const double effective_confidence = logistic_confidence(effective_risk, config);
    result.surface_confidence_mean = effective_level / config.surface_levels[levels - 1];
    result.surface_confidence_edge = effective_confidence;
    result.surface_effective_level = effective_level;
    result.surface_confidence_risk = effective_risk;

    SurfaceScreen screen;
    screen.level = std::max(effective_level, config.surface_levels[0]);
    double mean_radius = 0.0;
    double maximum_radius = 0.0;
    for (int theta_index = 0; theta_index < theta_count; ++theta_index) {
        const double theta = TWOPI * theta_index / theta_count;
        std::array<double, 25> polynomial;
        ray_polynomial_phi0(psi, config, theta, polynomial);
        const double radius = solve_ray_radius(polynomial, screen.level, config);
        mean_radius += radius / theta_count;
        maximum_radius = std::max(maximum_radius, radius);
    }
    screen.radius_mean = mean_radius;
    screen.radius_max = maximum_radius;
    screen.relative_drift_p95 = effective_risk;
    screen.one_period_relative_drift_p95 = effective_risk;
    screen.drift_p95 = effective_risk * mean_radius;
    screen.long_trace_periods_completed = config.surface_confidence_periods;
    screen.long_verified = false;
    screen.stable = effective_level >= config.surface_levels[0];
    screen.strict = screen.stable;
    screen.verified = true;
    screens.assign(1, screen);
    result.surface_confidence_edge = effective_confidence;
    result.timings[SGPU_SCORE_TIME_SURFACE_CONFIDENCE] += seconds_since(substage_started);
    return true;
}

bool verify_surface_long_horizon(
    void* field,
    const AxisData& axis,
    const PsiData& psi,
    int nfp,
    const SgpuScoreConfig& config,
    SurfaceScreen& screen,
    double* timings
) {
    screen.long_verified = true;
    screen.long_trace_periods_completed = 1;
    if (!screen.strict || config.surface_long_trace_periods <= 1) return true;

    const int theta_count = config.surface_theta_count;
    std::vector<double> R(theta_count), Z(theta_count), R_end, Z_end;
    for (int theta_index = 0; theta_index < theta_count; ++theta_index) {
        const double theta = TWOPI * theta_index / theta_count;
        std::array<double, 25> polynomial;
        ray_polynomial_phi0(psi, config, theta, polynomial);
        const double radius = solve_ray_radius(polynomial, screen.level, config);
        R[theta_index] = axis.R[0] + radius * std::cos(theta);
        Z[theta_index] = axis.Z[0] + radius * std::sin(theta);
    }

    double maximum_relative_drift = screen.relative_drift_p95;
    for (int period = 1; period <= config.surface_long_trace_periods; ++period) {
        auto substage_started = Clock::now();
        if (!trace_map(field, R, Z, nfp, config.surface_trace_steps, false, R_end, Z_end)) {
            return false;
        }
        timings[SGPU_SCORE_TIME_SURFACE_LONG_TRACE] += seconds_since(substage_started);
        substage_started = Clock::now();
        std::vector<double> distances;
        distances.reserve(theta_count);
        for (int theta_index = 0; theta_index < theta_count; ++theta_index) {
            double value, gR, gZ, gPhi;
            evaluate_psi_host(
                psi, axis, config, nfp, R_end[theta_index], Z_end[theta_index], TWOPI / nfp,
                value, gR, gZ, gPhi
            );
            const double gradient_norm = std::sqrt(
                gR * gR + gZ * gZ + std::pow(gPhi / R_end[theta_index], 2.0)
            );
            const double distance = std::abs(value - screen.level) /
                std::max(gradient_norm, 1.0e-14);
            distances.push_back(std::isfinite(distance)
                ? distance : std::numeric_limits<double>::infinity());
        }
        const double relative_drift = percentile(distances, 0.95) /
            std::max(screen.radius_mean, 1.0e-14);
        maximum_relative_drift = std::max(maximum_relative_drift, relative_drift);
        screen.long_trace_periods_completed = period;
        R.swap(R_end);
        Z.swap(Z_end);
        timings[SGPU_SCORE_TIME_SURFACE_LONG_REDUCE] += seconds_since(substage_started);
        if (maximum_relative_drift > config.surface_long_trace_relative_tolerance) break;
    }
    screen.relative_drift_p95 = maximum_relative_drift;
    screen.drift_p95 = maximum_relative_drift * screen.radius_mean;
    screen.strict = screen.strict &&
        maximum_relative_drift <= config.surface_long_trace_relative_tolerance &&
        screen.long_trace_periods_completed == config.surface_long_trace_periods;
    return true;
}

bool validate_config(const SgpuScoreConfig& config, std::string& reason) {
    if (config.abi_version != SGPU_SCORE_ABI_VERSION || config.struct_size != sizeof(SgpuScoreConfig)) {
        reason = "score config ABI version or size mismatch";
        return false;
    }
    if (config.device_id < 0 || config.segments_per_coil <= 0 ||
        config.axis_grid < 8 || config.axis_fallback_grid < config.axis_grid ||
        config.axis_fallback_max_nfp < 1 ||
        config.axis_sample_count < 16 || config.psi_poly_degree < 2 ||
        config.psi_poly_degree > 24 || config.psi_m_tor < 0 || config.psi_m_tor > 32 ||
        (config.psi_solver_mode != 1 && config.psi_solver_mode != 2) ||
        (config.psi_precision_mode != 1 && config.psi_precision_mode != 2) ||
        config.surface_level_count <= 0 ||
        config.surface_level_count > SGPU_SCORE_MAX_SURFACE_LEVELS ||
        config.surface_theta_count < 16 || config.surface_long_trace_periods < 1 ||
        config.surface_long_trace_periods > 64 ||
        !(config.surface_long_trace_relative_tolerance > 0.0) ||
        config.volume_point_count <= 0 ||
        config.alpha_fit_point_count <= 0 ||
        config.alpha_fit_point_count > config.volume_point_count ||
        (config.alpha_solver_mode != 1 && config.alpha_solver_mode != 2) ||
        !(config.score_surface_inverse_aspect_saturation > 0.0) ||
        !(config.score_qh_iota_threshold > 0.0) ||
        !(config.score_qh_iota_power > 0.0) ||
        config.score_volume_qs_size_floor < 0.0 ||
        config.score_volume_qs_size_floor > 1.0 ||
        config.score_volume_qs_iota_floor < 0.0 ||
        config.score_volume_qs_iota_floor > 1.0 ||
        config.score_qh_total_iota_floor < 0.0 ||
        config.score_qh_total_iota_floor > 1.0 ||
        config.score_qh_total_helicity_floor < 0.0 ||
        config.score_qh_total_helicity_floor > 1.0 ||
        !(config.score_qh_helicity_good > config.score_qh_helicity_bad) ||
        config.axis_topology_margin < 0.0 || config.axis_topology_margin >= 2.0 ||
        config.score_qh_helicity_exploration_fraction < 0.0 ||
        config.score_qh_helicity_exploration_fraction > 1.0 ||
        config.surface_selection_mode < 0 || config.surface_selection_mode > 1 ||
        config.surface_confidence_periods < 1 || config.surface_confidence_periods > 16 ||
        config.surface_flux_bisection_iters < 0 || config.surface_flux_bisection_iters > 12 ||
        !(config.surface_confidence_drift_center > 0.0) ||
        !(config.surface_confidence_drift_temperature > 0.0) ||
        !(config.surface_confidence_smoothmax_temperature > 0.0) ||
        config.surface_confidence_minimum < 0.0 || config.surface_confidence_minimum > 1.0 ||
        config.axis_hint_enabled < 0 || config.axis_hint_enabled > 1 ||
        config.axis_hint_require_continuation < 0 || config.axis_hint_require_continuation > 1 ||
        !(config.axis_hint_max_distance > 0.0)) {
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
    const double stability_margin = axis_topology_stability_margin(axis.selected);
    const double topology = !axis.selected.elliptic
        ? 0.1
        : config.axis_topology_margin > 0.0
            ? q_saturating_up(stability_margin, config.axis_topology_margin)
            : 1.0;
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
    if (std::isfinite(result.surface_drift_relative_p95)) {
        minimum_drift = result.surface_drift_relative_p95;
    }
    const double size = q_saturating_up(
        result.surface_inverse_aspect_ratio, config.score_surface_inverse_aspect_saturation
    );
    result.score_surface_size = size;
    if (config.surface_selection_mode == 1) {
        const double selected_level = std::isfinite(result.surface_level)
            ? result.surface_level : result.surface_effective_level;
        const double extent = q_saturating_up(
            selected_level, config.surface_levels[config.surface_level_count - 1]
        );
        // The selected level already combines short-horizon confidence with the
        // continuously calibrated flux boundary. Do not reuse risk measured at
        // the larger proposal after flux calibration has moved the edge inward.
        result.components[SGPU_SCORE_COMPONENT_SURFACE] = blend({
            {0.65, size}, {0.35, extent},
        });
        result.components[SGPU_SCORE_COMPONENT_COIL] = coil_component(coil);
        return;
    }
    const double drift = q_down(minimum_drift, config.score_surface_drift_scale, 1.0, 0.15);
    const double count = q_up(strict_count, 2.0, 1.0);
    result.components[SGPU_SCORE_COMPONENT_SURFACE] = blend({{0.65, size}, {0.25, drift}, {0.10, count}});
    result.components[SGPU_SCORE_COMPONENT_COIL] = coil_component(coil);
}

void finalize_score(const SgpuScoreConfig& config, SgpuScoreResult& result) {
    if (result.status == SGPU_SCORE_INTERNAL_ERROR) return;
    if (result.stage_completed < SCORE_STAGE_ALPHA) {
        result.components[SGPU_SCORE_COMPONENT_COORDINATE] = 0.08;
    }
    if (result.stage_completed < SCORE_STAGE_QS) {
        result.components[SGPU_SCORE_COMPONENT_VOLUME_QS] = 0.04;
    }
    if (result.stage_completed < SCORE_STAGE_ALPHA) {
        result.components[SGPU_SCORE_COMPONENT_IOTA] = 0.04;
    }
    double total_weight = 0.0;
    double weighted_score = 0.0;
    const double iota_score = clip01(result.components[SGPU_SCORE_COMPONENT_IOTA]);
    for (int component = 0; component < SGPU_SCORE_COMPONENT_COUNT; ++component) {
        const double value = clip01(result.components[component]);
        total_weight += std::max(config.score_weights[component], 0.0);
        weighted_score += std::max(config.score_weights[component], 0.0) * value;
        result.components[component] = 100.0 * value;
    }
    result.score_before_qh_iota_gate =
        total_weight > 0.0 ? 100.0 * clip01(weighted_score / total_weight) : 0.0;
    const bool qh_target = config.target_M != 0 && config.target_N != 0;
    result.score_qh_total_iota_factor = qh_target
        ? config.score_qh_total_iota_floor +
            (1.0 - config.score_qh_total_iota_floor) * iota_score
        : 1.0;
    const double helicity_position = clip01(
        (result.score_qh_helicity_advantage - config.score_qh_helicity_bad) /
        (config.score_qh_helicity_good - config.score_qh_helicity_bad)
    );
    const double helicity_linear = clip01(
        result.score_qh_helicity_advantage / config.score_qh_helicity_good
    );
    const double helicity_window =
        helicity_position * helicity_position * (3.0 - 2.0 * helicity_position);
    result.score_qh_helicity_quality = qh_target
        ? config.score_qh_helicity_exploration_fraction * helicity_linear +
            (1.0 - config.score_qh_helicity_exploration_fraction) * helicity_window
        : 1.0;
    result.score_qh_total_helicity_factor = qh_target
        ? config.score_qh_total_helicity_floor +
            (1.0 - config.score_qh_total_helicity_floor) *
                result.score_qh_helicity_quality
        : 1.0;
    result.score = result.score_before_qh_iota_gate *
        result.score_qh_total_iota_factor * result.score_qh_total_helicity_factor;
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
    {
        const float u = radius / a_scale;
        float value = 0.0f;
        float power = u * u;
        for (int degree_now = 2; degree_now <= degree; ++degree_now) {
            if (degree_now > 2) power *= u;
            value += polynomial[degree_now] * power;
        }
        final_residual = value - level;
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
    const float* __restrict__ boundary_residuals,
    int n_phi,
    int n_theta,
    int n_radial,
    int nfp,
    float rho_min,
    float edge_level,
    float boundary_tolerance,
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
    volume_weight[index] = R * boundary * boundary;
    flux_derivative_out[index] = flux_derivative;
    const float lower = edge_level * rho_min * rho_min;
    valid[index] = static_cast<unsigned char>(
        isfinite(s_value) && isfinite(physical_rho) && R > 0.0f &&
        boundary_residuals[ray] <= boundary_tolerance &&
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
    result.volume_candidate_count = candidate_count;
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
        d_boundary_radii.data(), d_boundary_residuals.data(), n_phi, n_theta, n_radial, nfp,
        static_cast<float>(config.volume_rho_min), static_cast<float>(flux.edge_level),
        static_cast<float>(config.flux_boundary_tolerance),
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
    result.volume_available_count = available_count;
    result.volume_valid_fraction = static_cast<double>(available_count) /
        std::max(candidate_count, 1);
    const int minimum_count = std::max(
        std::max(config.alpha_fit_point_count, config.volume_point_count),
        static_cast<int>(std::ceil(0.95 * candidate_count))
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
    int mode_count = 0;
    int fit_count = 0;
    int qr_rows = 0;
    double residual_norm = 0.0;
    double rhs_norm = 0.0;
    double weight_norm2 = 0.0;
    double normal_numerator = 0.0;
    double field_denominator = 0.0;
    ClebschModesNative modes;
    DeviceBuffer<float> qr_matrix;
    DeviceBuffer<float> matrix_reference;
    DeviceBuffer<float> rhs_reference;
    DeviceBuffer<float> scales;
    DeviceBuffer<float> scaled_solution;
    DeviceBuffer<float> residual;
    DeviceBuffer<float> b_theta;
    DeviceBuffer<float> b_phi;
    DeviceBuffer<float> weights;
    DeviceBuffer<float> rho;
    DeviceBuffer<float> theta;
    DeviceBuffer<float> phi;
    bool adjoint_ready = false;
};

struct FixedFrontG2Cache {
    AxisData axis;
    DeviceVolumePoints points;
    DeviceBuffer<float> B;
    DeviceBuffer<float> grad_B;
    AlphaFitNative alpha;
    std::array<double, 9> qs_sums{};
    double G = 0.0;
    void* field = nullptr;
    bool ready = false;
};

thread_local FixedFrontG2Cache* g_active_g2_cache = nullptr;
thread_local int g_active_gradient_group = 0;

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
    int info = 0;
    cublasStatus_t blas_status = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t solver_status = CUSOLVER_STATUS_SUCCESS;
    float* d_scaled_solution = nullptr;
    DeviceBuffer<float> d_normal_matrix;
    DeviceBuffer<float> d_normal_rhs;
    DeviceBuffer<float> d_normal_work;
    if (config.alpha_solver_mode == 1) {
        if (!d_normal_matrix.allocate(static_cast<size_t>(column_count) * column_count) ||
            !d_normal_rhs.allocate(column_count)) {
            cublasDestroy(blas);
            cusolverDnDestroy(solver);
            fail_result(&result, "alpha normal-equation allocation failed");
            return false;
        }
        const float one = 1.0f;
        const float zero = 0.0f;
        blas_status = cublasSgemm(
            blas, CUBLAS_OP_T, CUBLAS_OP_N, column_count, column_count, qr_rows,
            &one, d_matrix.data(), qr_rows, d_matrix.data(), qr_rows,
            &zero, d_normal_matrix.data(), column_count
        );
        if (blas_status == CUBLAS_STATUS_SUCCESS) {
            blas_status = cublasSgemv(
                blas, CUBLAS_OP_T, qr_rows, column_count, &one,
                d_matrix.data(), qr_rows, d_rhs.data(), 1,
                &zero, d_normal_rhs.data(), 1
            );
        }
        int work_size = 0;
        if (blas_status == CUBLAS_STATUS_SUCCESS) {
            solver_status = cusolverDnSpotrf_bufferSize(
                solver, CUBLAS_FILL_MODE_LOWER, column_count,
                d_normal_matrix.data(), column_count, &work_size
            );
        }
        if (solver_status == CUSOLVER_STATUS_SUCCESS && !d_normal_work.allocate(work_size)) {
            solver_status = CUSOLVER_STATUS_ALLOC_FAILED;
        }
        if (solver_status == CUSOLVER_STATUS_SUCCESS && blas_status == CUBLAS_STATUS_SUCCESS) {
            solver_status = cusolverDnSpotrf(
                solver, CUBLAS_FILL_MODE_LOWER, column_count,
                d_normal_matrix.data(), column_count, d_normal_work.data(), work_size,
                d_info.data()
            );
            cudaMemcpy(&info, d_info.data(), sizeof(int), cudaMemcpyDeviceToHost);
        }
        if (solver_status == CUSOLVER_STATUS_SUCCESS && info == 0) {
            solver_status = cusolverDnSpotrs(
                solver, CUBLAS_FILL_MODE_LOWER, column_count, 1,
                d_normal_matrix.data(), column_count, d_normal_rhs.data(), column_count,
                d_info.data()
            );
            cudaMemcpy(&info, d_info.data(), sizeof(int), cudaMemcpyDeviceToHost);
        }
        if (solver_status != CUSOLVER_STATUS_SUCCESS || info != 0 ||
            blas_status != CUBLAS_STATUS_SUCCESS) {
            cublasDestroy(blas);
            cusolverDnDestroy(solver);
            fail_result(&result, "alpha FP32 Cholesky normal-equation solve failed");
            return false;
        }
        d_scaled_solution = d_normal_rhs.data();
    } else {
        int geqrf_work = 0;
        int ormqr_work = 0;
        solver_status = cusolverDnSgeqrf_bufferSize(
            solver, qr_rows, column_count, d_matrix.data(), qr_rows, &geqrf_work
        );
        if (solver_status == CUSOLVER_STATUS_SUCCESS) {
            solver_status = cusolverDnSormqr_bufferSize(
                solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, qr_rows, 1, column_count,
                d_matrix.data(), qr_rows, d_tau.data(), d_rhs.data(), qr_rows, &ormqr_work
            );
        }
        DeviceBuffer<float> d_work(std::max(geqrf_work, ormqr_work));
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
        if (solver_status == CUSOLVER_STATUS_SUCCESS && info == 0) {
            blas_status = cublasStrsv(
                blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                column_count, d_matrix.data(), qr_rows, d_rhs.data(), 1
            );
        }
        if (solver_status != CUSOLVER_STATUS_SUCCESS || info != 0 ||
            blas_status != CUBLAS_STATUS_SUCCESS) {
            cublasDestroy(blas);
            cusolverDnDestroy(solver);
            fail_result(&result, "alpha FP32 QR solve failed");
            return false;
        }
        d_scaled_solution = d_rhs.data();
    }
    unscale_alpha_solution_kernel<<<(column_count + threads - 1) / threads, threads>>>(
        d_scaled_solution, d_scales.data(), column_count, d_solution.data()
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
        d_matrix_reference.data(), fit_count, d_scaled_solution, 1, &zero, d_prediction.data(), 1
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
    fit.residual_norm = residual_norm;
    fit.rhs_norm = rhs_norm;
    fit.iota_coefficients.resize(config.iota_degree + 1);
    if (!cuda_stage_ok(cudaMemcpy(
            fit.iota_coefficients.data(), d_solution.data() + mode_count,
            fit.iota_coefficients.size() * sizeof(float), cudaMemcpyDeviceToHost
        ), result, "alpha iota copy")) {
        cublasDestroy(blas);
        cusolverDnDestroy(solver);
        return false;
    }
    if (g_active_gradient_group >= 3) {
        DeviceBuffer<float> d_saved_scaled_solution(column_count);
        if (!d_saved_scaled_solution.data() ||
            !cuda_stage_ok(cudaMemcpy(
                d_saved_scaled_solution.data(), d_scaled_solution,
                column_count * sizeof(float), cudaMemcpyDeviceToDevice
            ), result, "alpha scaled solution cache")) {
            cublasDestroy(blas);
            cusolverDnDestroy(solver);
            return false;
        }
        fit.mode_count = mode_count;
        fit.fit_count = fit_count;
        fit.qr_rows = qr_rows;
        fit.weight_norm2 = diagnostics[0];
        fit.normal_numerator = diagnostics[1];
        fit.field_denominator = diagnostics[2];
        fit.modes = modes;
        fit.qr_matrix = std::move(d_matrix);
        fit.matrix_reference = std::move(d_matrix_reference);
        fit.rhs_reference = std::move(d_rhs_reference);
        fit.scales = std::move(d_scales);
        fit.scaled_solution = std::move(d_saved_scaled_solution);
        fit.residual = std::move(d_prediction);
        fit.b_theta = std::move(d_b_theta);
        fit.b_phi = std::move(d_b_phi);
        fit.weights = std::move(d_weights);
        fit.rho = std::move(d_rho);
        fit.theta = std::move(d_theta);
        fit.phi = std::move(d_phi);
        fit.adjoint_ready = true;
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
    int nfp,
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
    const double f_c_qa = static_cast<double>(iota) * A - G * C;
    const double f_c_qp = -static_cast<double>(nfp) * A;
    const double normalized = f_c /
        fmax(static_cast<double>(magnitude) * magnitude * magnitude, 1.0e-30);
    const double normalized_qa = f_c_qa /
        fmax(static_cast<double>(magnitude) * magnitude * magnitude, 1.0e-30);
    const double normalized_qp = f_c_qp /
        fmax(static_cast<double>(magnitude) * magnitude * magnitude, 1.0e-30);
    const double weight = volume_weight[point];
    absolute_normalized[point] = static_cast<float>(fabs(normalized));
    atomic_add_double(sums, weight);
    atomic_add_double(sums + 1, weight * normalized * normalized);
    atomic_add_double(sums + 4, weight * weight);
    atomic_add_double(sums + 7, weight * normalized_qa * normalized_qa);
    atomic_add_double(sums + 8, weight * normalized_qp * normalized_qp);
    if (rho[point] >= edge_rho_threshold) {
        atomic_add_double(sums + 2, weight);
        atomic_add_double(sums + 3, weight * normalized * normalized);
        atomic_add_double(sums + 5, weight * weight);
        atomic_add_double(sums + 6, 1.0);
    }
}

__global__ void compute_qs_point_adjoint_kernel(
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
    int nfp,
    double G,
    float edge_rho_threshold,
    double weight_sum,
    double edge_weight_sum,
    double target_rms,
    double target_edge_rms,
    double qa_rms,
    double qp_rms,
    double d_target_error,
    double d_target_edge_error,
    double d_qa_error,
    double d_qp_error,
    float* __restrict__ adj_B,
    float* __restrict__ adj_grad_B,
    double* __restrict__ adj_G
) {
    const int point = blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= point_count) return;
    float b[3];
    float magnitude2 = 0.0f;
    for (int component = 0; component < 3; ++component) {
        b[component] = B[3 * point + component];
        magnitude2 += b[component] * b[component];
    }
    const float magnitude = sqrtf(fmaxf(magnitude2, 1.0e-30f));
    float q[3] = {};
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        for (int component = 0; component < 3; ++component) {
            q[coordinate] += grad_B[9 * point + 3 * component + coordinate] * b[component];
        }
        q[coordinate] /= magnitude;
    }
    float p[3];
    for (int component = 0; component < 3; ++component) {
        p[component] = flux_derivative[point] * grad_s[3 * point + component];
    }
    const float cross[3] = {
        b[1] * p[2] - b[2] * p[1],
        b[2] * p[0] - b[0] * p[2],
        b[0] * p[1] - b[1] * p[0],
    };
    const float A = cross[0] * q[0] + cross[1] * q[1] + cross[2] * q[2];
    const float C = b[0] * q[0] + b[1] * q[1] + b[2] * q[2];
    const float u = rho[point] * rho[point];
    float iota = 0.0f;
    float power = 1.0f;
    for (int degree = 0; degree <= iota_degree; ++degree) {
        iota += iota_coefficients[degree] * power;
        power *= u;
    }
    const double k_target = helicity_M * static_cast<double>(iota) - helicity_N;
    const double f_target = k_target * A - helicity_M * G * C;
    const double f_qa = static_cast<double>(iota) * A - G * C;
    const double f_qp = -static_cast<double>(nfp) * A;
    const double magnitude3 = fmax(static_cast<double>(magnitude) * magnitude * magnitude, 1.0e-30);
    const double r_target = f_target / magnitude3;
    const double r_qa = f_qa / magnitude3;
    const double r_qp = f_qp / magnitude3;
    const double weight = volume_weight[point];
    const double target_norm = fmax(
        hypot(static_cast<double>(helicity_M), static_cast<double>(helicity_N)), 1.0
    );
    const double qp_norm = nfp > 1 ? static_cast<double>(nfp) : 1.0;
    double lambda_target = 0.0;
    if (target_rms > 1.0e-30 && weight_sum > 0.0) {
        lambda_target += d_target_error * weight * r_target /
            (weight_sum * target_rms * target_norm);
    }
    if (rho[point] >= edge_rho_threshold && target_edge_rms > 1.0e-30 && edge_weight_sum > 0.0) {
        lambda_target += d_target_edge_error * weight * r_target /
            (edge_weight_sum * target_edge_rms * target_norm);
    }
    const double lambda_qa = qa_rms > 1.0e-30 && weight_sum > 0.0
        ? d_qa_error * weight * r_qa / (weight_sum * qa_rms)
        : 0.0;
    const double lambda_qp = qp_rms > 1.0e-30 && weight_sum > 0.0
        ? d_qp_error * weight * r_qp / (weight_sum * qp_rms * qp_norm)
        : 0.0;

    const double adj_A =
        (lambda_target * k_target + lambda_qa * iota - lambda_qp * nfp) / magnitude3;
    const double adj_C = (-lambda_target * helicity_M * G - lambda_qa * G) / magnitude3;
    double adj_magnitude = -3.0 * (
        lambda_target * f_target + lambda_qa * f_qa + lambda_qp * f_qp
    ) / (magnitude3 * magnitude);
    const double point_adj_G =
        (-lambda_target * helicity_M * C - lambda_qa * C) / magnitude3;
    double adj_b[3] = {
        adj_A * (p[1] * q[2] - p[2] * q[1]) + adj_C * q[0],
        adj_A * (p[2] * q[0] - p[0] * q[2]) + adj_C * q[1],
        adj_A * (p[0] * q[1] - p[1] * q[0]) + adj_C * q[2],
    };
    const double adj_q[3] = {
        adj_A * cross[0] + adj_C * b[0],
        adj_A * cross[1] + adj_C * b[1],
        adj_A * cross[2] + adj_C * b[2],
    };
    double adj_y[3];
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        adj_y[coordinate] = adj_q[coordinate] / magnitude;
        adj_magnitude -= adj_q[coordinate] * q[coordinate] / magnitude;
    }
    for (int component = 0; component < 3; ++component) {
        for (int coordinate = 0; coordinate < 3; ++coordinate) {
            adj_b[component] +=
                grad_B[9 * point + 3 * component + coordinate] * adj_y[coordinate];
            adj_grad_B[9 * point + 3 * component + coordinate] =
                static_cast<float>(b[component] * adj_y[coordinate]);
        }
        adj_b[component] += adj_magnitude * b[component] / magnitude;
        adj_B[3 * point + component] = static_cast<float>(adj_b[component]);
    }
    atomic_add_double(adj_G, point_adj_G);
}

__global__ void compute_qs_iota_adjoint_kernel(
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
    double weight_sum,
    double edge_weight_sum,
    double target_rms,
    double target_edge_rms,
    double qa_rms,
    double d_target_error,
    double d_target_edge_error,
    double d_qa_error,
    double* __restrict__ adj_iota
) {
    const int point = blockIdx.x * blockDim.x + threadIdx.x;
    if (point >= point_count) return;
    float b[3];
    float magnitude2 = 0.0f;
    for (int component = 0; component < 3; ++component) {
        b[component] = B[3 * point + component];
        magnitude2 += b[component] * b[component];
    }
    const float magnitude = sqrtf(fmaxf(magnitude2, 1.0e-30f));
    float q[3] = {};
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        for (int component = 0; component < 3; ++component) {
            q[coordinate] += grad_B[9 * point + 3 * component + coordinate] * b[component];
        }
        q[coordinate] /= magnitude;
    }
    float grad_psi[3];
    for (int component = 0; component < 3; ++component) {
        grad_psi[component] = flux_derivative[point] * grad_s[3 * point + component];
    }
    const float cross[3] = {
        b[1] * grad_psi[2] - b[2] * grad_psi[1],
        b[2] * grad_psi[0] - b[0] * grad_psi[2],
        b[0] * grad_psi[1] - b[1] * grad_psi[0],
    };
    const double A = cross[0] * q[0] + cross[1] * q[1] + cross[2] * q[2];
    const double C = b[0] * q[0] + b[1] * q[1] + b[2] * q[2];
    const float u = rho[point] * rho[point];
    float iota = 0.0f;
    float power = 1.0f;
    for (int degree = 0; degree <= iota_degree; ++degree) {
        iota += iota_coefficients[degree] * power;
        power *= u;
    }
    const double magnitude3 = fmax(static_cast<double>(magnitude) * magnitude * magnitude, 1.0e-30);
    const double target_residual =
        ((helicity_M * static_cast<double>(iota) - helicity_N) * A - helicity_M * G * C) /
        magnitude3;
    const double qa_residual = (static_cast<double>(iota) * A - G * C) / magnitude3;
    const double weight = volume_weight[point];
    const double target_norm = fmax(
        hypot(static_cast<double>(helicity_M), static_cast<double>(helicity_N)), 1.0
    );
    double lambda_target = 0.0;
    if (target_rms > 1.0e-30 && weight_sum > 0.0) {
        lambda_target += d_target_error * weight * target_residual /
            (weight_sum * target_rms * target_norm);
    }
    if (rho[point] >= edge_rho_threshold && target_edge_rms > 1.0e-30 && edge_weight_sum > 0.0) {
        lambda_target += d_target_edge_error * weight * target_residual /
            (edge_weight_sum * target_edge_rms * target_norm);
    }
    const double lambda_qa = qa_rms > 1.0e-30 && weight_sum > 0.0
        ? d_qa_error * weight * qa_residual / (weight_sum * qa_rms)
        : 0.0;
    const double adj_iota_value = A / magnitude3 *
        (helicity_M * lambda_target + lambda_qa);
    double radial_power = 1.0;
    for (int degree = 0; degree <= iota_degree; ++degree) {
        atomic_add_double(adj_iota + degree, adj_iota_value * radial_power);
        radial_power *= u;
    }
}

__global__ void alpha_residual_adjoint_kernel(
    const float* __restrict__ residual,
    const float* __restrict__ rhs,
    int count,
    float residual_norm,
    float rhs_norm,
    float adj_relative,
    float* __restrict__ adj_residual,
    float* __restrict__ adj_rhs
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count) return;
    const float safe_residual = fmaxf(residual_norm, 1.0e-20f);
    const float safe_rhs = fmaxf(rhs_norm, 1.0e-20f);
    const float value = adj_relative * residual[row] / (safe_residual * safe_rhs);
    adj_residual[row] = value;
    adj_rhs[row] = -value - adj_relative * residual_norm * rhs[row] /
        (safe_rhs * safe_rhs * safe_rhs);
}

__global__ void add_alpha_solution_adjoint_kernel(
    const float* __restrict__ adj_solution,
    const float* __restrict__ scales,
    int count,
    float* __restrict__ adj_scaled_solution
) {
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column < count) {
        adj_scaled_solution[column] += adj_solution[column] / scales[column];
    }
}

__global__ void alpha_column_dot_kernel(
    const float* __restrict__ scaled_matrix,
    int row_count,
    const float* __restrict__ adj_residual,
    const float* __restrict__ residual,
    const float* __restrict__ multiplier,
    const float* __restrict__ matrix_multiplier,
    const float* __restrict__ scaled_solution,
    float* __restrict__ dot_adjoint_matrix
) {
    extern __shared__ double shared[];
    const int column = blockIdx.x;
    double sum = 0.0;
    const float lambda = multiplier[column];
    const float solution = scaled_solution[column];
    for (int row = threadIdx.x; row < row_count; row += blockDim.x) {
        const float matrix_value = scaled_matrix[static_cast<size_t>(column) * row_count + row];
        const float adjoint = adj_residual[row] * solution - residual[row] * lambda -
            matrix_multiplier[row] * solution;
        sum += static_cast<double>(adjoint) * matrix_value;
    }
    shared[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) dot_adjoint_matrix[column] = static_cast<float>(shared[0]);
}

__global__ void alpha_row_adjoint_kernel(
    const float* __restrict__ scaled_matrix,
    int row_count,
    int column_count,
    const float* __restrict__ adj_residual,
    const float* __restrict__ residual,
    const float* __restrict__ adj_rhs,
    const float* __restrict__ multiplier,
    const float* __restrict__ matrix_multiplier,
    const float* __restrict__ scaled_solution,
    const float* __restrict__ adj_solution,
    const float* __restrict__ scales,
    const float* __restrict__ dot_adjoint_matrix,
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
    float* __restrict__ adj_weight,
    float* __restrict__ adj_b_theta,
    float* __restrict__ adj_b_phi
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= row_count) return;
    const float bt = b_theta[row];
    const float bp = b_phi[row];
    const float weight = weights[row];
    float weight_adjoint = -adj_rhs[row] * bt;
    float theta_adjoint = -adj_rhs[row] * weight;
    float phi_adjoint = 0.0f;
    for (int column = 0; column < column_count; ++column) {
        const float scale = scales[column];
        const float matrix_value = scaled_matrix[static_cast<size_t>(column) * row_count + row];
        const float matrix_adjoint =
            adj_residual[row] * scaled_solution[column] -
            residual[row] * multiplier[column] -
            matrix_multiplier[row] * scaled_solution[column];
        const float scale_adjoint =
            -adj_solution[column] * scaled_solution[column] / (scale * scale) -
            dot_adjoint_matrix[column] / scale;
        const float design_adjoint = matrix_adjoint / scale + matrix_value * scale_adjoint;
        if (column < mode_count) {
            const int m = mode_m[column];
            const int n = mode_n[column];
            const float radial = evaluate_radial_polynomial(
                radial_coefficients + static_cast<size_t>(column) * (radial_order + 1),
                radial_order,
                rho[row]
            );
            float sine, cosine;
            sincosf(m * theta[row] - n * nfp * phi[row], &sine, &cosine);
            const float derivative_theta = mode_kind[column] == 0
                ? -m * radial * sine : m * radial * cosine;
            const float derivative_phi = mode_kind[column] == 0
                ? n * nfp * radial * sine : -n * nfp * radial * cosine;
            weight_adjoint += design_adjoint * (derivative_theta * bt + derivative_phi * bp);
            theta_adjoint += design_adjoint * weight * derivative_theta;
            phi_adjoint += design_adjoint * weight * derivative_phi;
        } else {
            const int power_index = column - mode_count;
            if (power_index <= iota_degree) {
                float radial_power = 1.0f;
                const float u = rho[row] * rho[row];
                for (int degree = 0; degree < power_index; ++degree) radial_power *= u;
                weight_adjoint -= design_adjoint * radial_power * bp;
                phi_adjoint -= design_adjoint * weight * radial_power;
            }
        }
    }
    adj_weight[row] = weight_adjoint;
    adj_b_theta[row] = theta_adjoint;
    adj_b_phi[row] = phi_adjoint;
}

__global__ void alpha_B_adjoint_kernel(
    const float* __restrict__ B,
    const float* __restrict__ grad_s,
    const float* __restrict__ grad_theta,
    const float* __restrict__ grad_phi,
    int point_count,
    int fit_count,
    const float* __restrict__ weights,
    const float* __restrict__ adj_weight,
    const float* __restrict__ adj_b_theta,
    const float* __restrict__ adj_b_phi,
    float weight_scale,
    float adj_weight_dot_weight,
    float adj_normal_relative,
    float normal_relative,
    float normal_numerator,
    float field_denominator,
    float* __restrict__ adj_B
) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= fit_count) return;
    const int point = min(
        point_count - 1,
        static_cast<int>((static_cast<long long>(row) * point_count) / fit_count)
    );
    float b[3];
    float gs[3];
    float gt[3];
    float gp[3];
    float magnitude2 = 0.0f;
    float gs2 = 0.0f;
    for (int component = 0; component < 3; ++component) {
        b[component] = B[3 * point + component];
        gs[component] = grad_s[3 * point + component];
        gt[component] = grad_theta[3 * point + component];
        gp[component] = grad_phi[3 * point + component];
        magnitude2 += b[component] * b[component];
        gs2 += gs[component] * gs[component];
    }
    float projection[3];
    float projection_dot_gs = 0.0f;
    for (int component = 0; component < 3; ++component) {
        projection[component] =
            adj_b_theta[row] * gt[component] + adj_b_phi[row] * gp[component];
        projection_dot_gs += projection[component] * gs[component];
    }
    const float inverse_gs2 = 1.0f / fmaxf(gs2, 1.0e-30f);
    const float normalized_weight = weights[row];
    const float base_weight = normalized_weight / weight_scale;
    const float base_weight_adjoint = weight_scale * (
        adj_weight[row] - normalized_weight * adj_weight_dot_weight / fit_count
    );
    const float normal_coefficient =
        (b[0] * gs[0] + b[1] * gs[1] + b[2] * gs[2]) * inverse_gs2;
    for (int component = 0; component < 3; ++component) {
        float value = projection[component] - projection_dot_gs * inverse_gs2 * gs[component];
        value -= base_weight_adjoint * base_weight * b[component] /
            fmaxf(magnitude2, 1.0e-30f);
        if (adj_normal_relative != 0.0f && normal_relative > 1.0e-20f &&
            field_denominator > 1.0e-20f && normal_numerator >= 0.0f) {
            value += adj_normal_relative * (
                normal_coefficient * gs[component] /
                    (normal_relative * field_denominator) -
                normal_relative * b[component] / field_denominator
            );
        }
        adj_B[3 * point + component] = value;
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
    double edge_toroidal_flux,
    const SgpuScoreConfig& config,
    SgpuScoreResult& result,
    std::array<double, 9>* raw_sums
) {
    DeviceBuffer<float> d_iota;
    DeviceBuffer<float> d_absolute(points.count);
    DeviceBuffer<double> d_sums(9);
    if (!copy_to_device(d_iota, alpha.iota_coefficients) || !d_absolute.data() || !d_sums.data() ||
        !cuda_stage_ok(cudaMemset(d_sums.data(), 0, d_sums.size() * sizeof(double)), result, "QS reduction clear")) {
        if (result.status != SGPU_SCORE_INTERNAL_ERROR) fail_result(&result, "QS allocation failed");
        return false;
    }
    double current_sum = 0.0;
    for (int index = 0; index < n_base_coils; ++index) current_sum += std::abs(currents_a[index]);
    if (!std::isfinite(edge_toroidal_flux) || edge_toroidal_flux == 0.0) {
        fail_result(&result, "QS metric requires nonzero signed toroidal flux");
        return false;
    }
    // The volume coordinates use radian angles, so G = mu0 I_link / (2 pi).
    const double G_magnitude = 2.0e-7 * 2.0 * nfp * current_sum;
    const double G = std::copysign(G_magnitude, edge_toroidal_flux);
    const float edge_threshold = static_cast<float>(
        config.volume_rho_min + (1.0 - config.volume_rho_min) *
        (config.radial_bin_count - 1.0) / config.radial_bin_count
    );
    constexpr int threads = 256;
    compute_qs_metric_kernel<<<(points.count + threads - 1) / threads, threads>>>(
        d_B, d_grad_B, points.grad_s.data(), points.flux_derivative.data(),
        points.rho.data(), points.volume_weight.data(), points.count,
        d_iota.data(), config.iota_degree, config.target_M, config.target_N,
        nfp, G, edge_threshold, d_absolute.data(), d_sums.data()
    );
    if (!cuda_stage_ok(cudaDeviceSynchronize(), result, "QS metric kernel")) return false;
    thrust::device_ptr<float> begin(d_absolute.data());
    thrust::sort(begin, begin + points.count);
    const int percentile_index = std::min(points.count - 1, static_cast<int>(std::floor(0.95 * (points.count - 1))));
    float p95 = 0.0f;
    std::array<double, 9> sums{};
    if (!cuda_stage_ok(cudaMemcpy(&p95, d_absolute.data() + percentile_index, sizeof(float), cudaMemcpyDeviceToHost), result, "QS p95 copy") ||
        !cuda_stage_ok(cudaMemcpy(sums.data(), d_sums.data(), sizeof(sums), cudaMemcpyDeviceToHost), result, "QS sums copy")) {
        return false;
    }
    result.qs_global_error = std::sqrt(sums[1] / std::max(sums[0], 1.0e-300));
    result.qs_edge_error = std::sqrt(sums[3] / std::max(sums[2], 1.0e-300));
    result.qs_qa_global_error = std::sqrt(sums[7] / std::max(sums[0], 1.0e-300));
    result.qs_qp_global_error_raw = std::sqrt(sums[8] / std::max(sums[0], 1.0e-300));
    const double target_helicity_norm = std::max(
        std::hypot(static_cast<double>(config.target_M), static_cast<double>(config.target_N)),
        1.0
    );
    const double qp_helicity_norm = std::max(std::abs(nfp), 1);
    result.qs_vacuum_G = G;
    result.qs_target_global_error_per_helicity = result.qs_global_error / target_helicity_norm;
    result.qs_target_edge_error_per_helicity = result.qs_edge_error / target_helicity_norm;
    result.qs_qa_global_error_per_helicity = result.qs_qa_global_error;
    result.qs_qp_global_error_per_helicity = result.qs_qp_global_error_raw / qp_helicity_norm;
    // Preserve the ABI-8 field semantics while exposing both explicit forms above.
    result.qs_qp_global_error = result.qs_qp_global_error_per_helicity;
    result.volume_weight_effective_fraction =
        sums[0] * sums[0] / std::max(points.count * sums[4], 1.0e-300);
    result.edge_weight_effective_fraction =
        sums[2] * sums[2] / std::max(sums[6] * sums[5], 1.0e-300);
    result.qs_abs_p95 = p95;
    result.qs_abs_p95_per_helicity = p95 / target_helicity_norm;
    if (raw_sums) *raw_sums = sums;
    return true;
}

bool run_downstream_gpu(
    void* field,
    const double* currents_a,
    int n_base_coils,
    int nfp,
    const AxisData& axis,
    const PsiData& psi,
    std::vector<SurfaceScreen>& screens,
    const SgpuScoreConfig& config,
    SgpuScoreResult& result,
    bool verify_long_horizon = true
) {
    DevicePsiData device_psi;
    if (!upload_psi_data(psi, axis, device_psi)) {
        fail_result(&result, "psi/axis upload for downstream pipeline failed");
        return false;
    }
    std::vector<SurfaceScreen*> candidates;
    for (auto& screen : screens) {
        if (screen.strict && screen.verified) candidates.push_back(&screen);
    }
    std::sort(candidates.begin(), candidates.end(), [](const auto* lhs, const auto* rhs) {
        return lhs->level > rhs->level;
    });
    auto started = Clock::now();
    FluxCalibrationNative flux;
    SurfaceScreen* selected_surface = nullptr;
    SurfaceScreen* closest_rejected_surface = nullptr;
    SurfaceScreen continuous_surface;
    bool flux_ok = false;
    auto calibrate_level = [&](double level, FluxCalibrationNative& trial) {
        ++result.flux_attempt_count;
        const auto calibration_started = Clock::now();
        const bool trial_ok = calibrate_flux_native(
            field, device_psi, axis, psi, nfp, level, config, trial, result
        );
        result.timings[SGPU_SCORE_TIME_FLUX_CALIBRATION] += seconds_since(calibration_started);
        return trial_ok;
    };
    if (config.surface_selection_mode == 1 && !candidates.empty()) {
        continuous_surface = *candidates.front();
        double high = continuous_surface.level;
        FluxCalibrationNative high_trial;
        const bool high_ok = calibrate_level(high, high_trial);
        if (result.status == SGPU_SCORE_INTERNAL_ERROR && result.error_message[0]) return false;
        if (high_ok) {
            flux = std::move(high_trial);
            flux_ok = true;
        } else {
            double low = std::numeric_limits<double>::quiet_NaN();
            FluxCalibrationNative low_trial;
            bool low_ok = false;
            for (int level_index = config.surface_level_count - 1;
                 level_index >= 0; --level_index) {
                const double probe = config.surface_levels[level_index];
                if (!(probe < high)) continue;
                FluxCalibrationNative probe_trial;
                const bool probe_ok = calibrate_level(probe, probe_trial);
                if (result.status == SGPU_SCORE_INTERNAL_ERROR && result.error_message[0]) {
                    return false;
                }
                if (probe_ok) {
                    low = probe;
                    low_trial = std::move(probe_trial);
                    low_ok = true;
                    break;
                }
                low_trial = std::move(probe_trial);
            }
            if (low_ok) {
                flux = std::move(low_trial);
                for (int iteration = 0; iteration < config.surface_flux_bisection_iters; ++iteration) {
                    const double middle = 0.5 * (low + high);
                    FluxCalibrationNative middle_trial;
                    const bool middle_ok = calibrate_level(middle, middle_trial);
                    if (result.status == SGPU_SCORE_INTERNAL_ERROR && result.error_message[0]) return false;
                    if (middle_ok) {
                        low = middle;
                        flux = std::move(middle_trial);
                    } else {
                        high = middle;
                    }
                }
                continuous_surface.level = low;
                flux_ok = true;
            } else {
                flux = std::move(low_trial);
                continuous_surface.level = config.surface_levels[0];
            }
        }
        selected_surface = &continuous_surface;
        result.surface_effective_level = continuous_surface.level;
    } else {
        for (SurfaceScreen* candidate : candidates) {
            const auto trace_started = Clock::now();
            if (verify_long_horizon) {
                if (!verify_surface_long_horizon(
                        field, axis, psi, nfp, config, *candidate, result.timings)) {
                    fail_from_backend(&result, "long-horizon surface trace");
                    return false;
                }
                result.timings[SGPU_SCORE_TIME_SURFACE_SCREEN] += seconds_since(trace_started);
            }
            if (!candidate->strict) {
                ++result.surface_long_trace_rejected_count;
                if (!closest_rejected_surface ||
                    candidate->relative_drift_p95 < closest_rejected_surface->relative_drift_p95) {
                    closest_rejected_surface = candidate;
                }
                continue;
            }
            FluxCalibrationNative trial;
            const bool trial_ok = calibrate_level(candidate->level, trial);
            if (result.status == SGPU_SCORE_INTERNAL_ERROR && result.error_message[0]) return false;
            if (!selected_surface || trial_ok) {
                selected_surface = candidate;
                flux = std::move(trial);
            }
            if (trial_ok) {
                flux_ok = true;
                break;
            }
        }
    }
    result.timings[SGPU_SCORE_TIME_FLUX] = seconds_since(started);
    if (!selected_surface) {
        if (closest_rejected_surface) {
            result.surface_level = closest_rejected_surface->level;
            result.surface_drift_relative_p95 = closest_rejected_surface->relative_drift_p95;
            result.surface_one_period_drift_relative_p95 =
                closest_rejected_surface->one_period_relative_drift_p95;
            result.surface_long_trace_periods_completed =
                closest_rejected_surface->long_trace_periods_completed;
        }
        result.status = SGPU_SCORE_DRIFT_REJECTED;
        return false;
    }
    result.surface_level = selected_surface->level;
    result.surface_drift_relative_p95 = selected_surface->relative_drift_p95;
    result.surface_one_period_drift_relative_p95 =
        selected_surface->one_period_relative_drift_p95;
    result.surface_long_trace_periods_completed =
        selected_surface->long_trace_periods_completed;
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
    if (!flux_ok) {
        result.status = SGPU_SCORE_FLUX_REJECTED;
        return false;
    }

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
    result.iota_min = std::numeric_limits<double>::infinity();
    result.iota_max = -std::numeric_limits<double>::infinity();
    const double u_min = config.volume_rho_min * config.volume_rho_min;
    for (int sample = 0; sample <= 256; ++sample) {
        const double u = u_min + (1.0 - u_min) * sample / 256.0;
        double iota = 0.0;
        double power = 1.0;
        for (float coefficient : alpha.iota_coefficients) {
            iota += coefficient * power;
            power *= u;
        }
        result.iota_min = std::min(result.iota_min, iota);
        result.iota_max = std::max(result.iota_max, iota);
    }
    result.stage_completed = SCORE_STAGE_ALPHA;
    started = Clock::now();
    std::array<double, 9> qs_sums{};
    if (!compute_qs_metric_native(
            points, d_B.data(), d_grad_B.data(), alpha, currents_a,
            n_base_coils, nfp, flux.edge_flux, config, result,
            g_active_g2_cache ? &qs_sums : nullptr)) {
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
    const bool qh_target = config.target_M != 0 && config.target_N != 0;
    const double iota_score = !qh_target ? 1.0 : std::pow(
        clip01(minimum_absolute_iota(result.iota_min, result.iota_max) /
               config.score_qh_iota_threshold),
        config.score_qh_iota_power
    );
    result.score_iota = iota_score;
    result.components[SGPU_SCORE_COMPONENT_IOTA] = iota_score;
    // f_C is linear in the helicity pair, so score its magnitude per unit (M, N).
    const double global_score = q_down(
        result.qs_target_global_error_per_helicity, config.score_qs_global_scale, 0.9
    );
    const double edge_score = q_down(
        result.qs_target_edge_error_per_helicity, config.score_qs_edge_scale, 0.9, global_score
    );
    const double residual_score = blend({{0.80, global_score}, {0.20, edge_score}});
    const double competitor_error = std::min(
        result.qs_qa_global_error_per_helicity, result.qs_qp_global_error_per_helicity
    );
    const double helicity_advantage = !qh_target ? 1.0 :
        competitor_error / std::max(
            result.qs_target_global_error_per_helicity + competitor_error, 1.0e-300
        );
    const double size_score = q_saturating_up(
        result.surface_inverse_aspect_ratio, config.score_surface_inverse_aspect_saturation
    );
    const double size_factor = config.score_volume_qs_size_floor +
        (1.0 - config.score_volume_qs_size_floor) * size_score;
    const double iota_factor = !qh_target ? 1.0 :
        config.score_volume_qs_iota_floor +
        (1.0 - config.score_volume_qs_iota_floor) * iota_score;
    result.score_surface_size = size_score;
    result.score_qs_residual = residual_score;
    result.score_qh_helicity_advantage = clip01(helicity_advantage);
    result.score_volume_qs_size_factor = size_factor;
    result.score_volume_qs_iota_factor = iota_factor;
    result.components[SGPU_SCORE_COMPONENT_VOLUME_QS] =
        residual_score * size_factor * iota_factor;
    result.status = SGPU_SCORE_OK;
    result.stage_completed = SCORE_STAGE_QS;
    if (g_active_g2_cache) {
        g_active_g2_cache->points = std::move(points);
        g_active_g2_cache->B = std::move(d_B);
        g_active_g2_cache->grad_B = std::move(d_grad_B);
        g_active_g2_cache->alpha = std::move(alpha);
        g_active_g2_cache->qs_sums = qs_sums;
        g_active_g2_cache->G = result.qs_vacuum_G;
        g_active_g2_cache->ready = true;
    }
    return true;
}

struct CoilComponentGradient {
    double value = 0.0;
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<double> current;
};

struct GeometrySample {
    Vec3d point;
    Vec3d first;
    Vec3d second;
    double curvature;
};

struct FullGeometryPoint {
    Vec3d point;
    int coil;
    int sample;
    int reflected;
    int period;
    int label;
};

Vec3d vec_sub(Vec3d left, Vec3d right) {
    return {left.x - right.x, left.y - right.y, left.z - right.z};
}

double vec_dot(Vec3d left, Vec3d right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

Vec3d vec_cross(Vec3d left, Vec3d right) {
    return {
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x,
    };
}

double vec_norm(Vec3d value) {
    return std::sqrt(std::max(vec_dot(value, value), 0.0));
}

void fourier_basis(int n_coeff, double t, int coefficient, double& value, double& first, double& second) {
    value = 0.0;
    first = 0.0;
    second = 0.0;
    if (coefficient == 0) {
        value = 1.0;
        return;
    }
    const int mode = (coefficient + 1) / 2;
    const double omega = TWOPI * mode;
    const double argument = omega * t;
    if (coefficient % 2 == 1) {
        value = std::sin(argument);
        first = omega * std::cos(argument);
        second = -omega * omega * std::sin(argument);
    } else {
        value = std::cos(argument);
        first = -omega * std::sin(argument);
        second = -omega * omega * std::cos(argument);
    }
}

Vec3d coordinate_vector(int coordinate, double value) {
    if (coordinate == 0) return {value, 0.0, 0.0};
    if (coordinate == 1) return {0.0, value, 0.0};
    return {0.0, 0.0, value};
}

double curvature_directional_derivative(
    const GeometrySample& sample,
    Vec3d d_first,
    Vec3d d_second
) {
    const double speed = vec_norm(sample.first);
    const Vec3d cross = vec_cross(sample.first, sample.second);
    const double cross_norm = vec_norm(cross);
    if (speed <= 1.0e-15 || cross_norm <= 1.0e-15) return 0.0;
    const Vec3d d_cross = vec_sub(
        vec_cross(d_first, sample.second),
        vec_cross(d_second, sample.first)
    );
    const double d_cross_norm = vec_dot(cross, d_cross) / cross_norm;
    const double d_speed = vec_dot(sample.first, d_first) / speed;
    return d_cross_norm / (speed * speed * speed) -
        3.0 * cross_norm * d_speed / (speed * speed * speed * speed);
}

Vec3d transformed_basis_vector(
    int coordinate,
    double basis,
    int reflected,
    int period,
    int nfp
) {
    Vec3d value = coordinate_vector(coordinate, basis);
    if (reflected) {
        value.y = -value.y;
        value.z = -value.z;
    }
    return rotate_z(value, TWOPI * static_cast<double>(period) / nfp);
}

double q_down_derivative(double value, double scale, double power) {
    if (!std::isfinite(value) || !(value > 0.0) || !(scale > 0.0)) return 0.0;
    const double x = value / scale;
    const double xp = std::pow(x, power);
    return -power * std::pow(x, power - 1.0) / (scale * (1.0 + xp) * (1.0 + xp));
}

double q_up_derivative(double value, double scale, double power) {
    if (!std::isfinite(value) || !(value > 0.0) || !(scale > 0.0)) return 0.0;
    const double ratio = std::pow(scale / value, power);
    return power * ratio / (value * (1.0 + ratio) * (1.0 + ratio));
}

bool compute_coil_component_gradient_impl(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    CoilComponentGradient& output,
    std::string& error
) {
    if (!coeffs_x || !coeffs_y || !coeffs_z || !currents_a ||
        n_base_coils <= 0 || n_coeff < 3 || (n_coeff % 2) == 0 || nfp <= 0) {
        error = "invalid coil-component gradient input";
        return false;
    }
    constexpr int samples = 160;
    const int parameter_count = n_base_coils * n_coeff;
    const int sample_count = n_base_coils * samples;
    std::vector<GeometrySample> geometry(sample_count);
    std::vector<double> curvatures(sample_count);
    std::vector<double> lengths(n_base_coils, 0.0);
    for (int coil = 0; coil < n_base_coils; ++coil) {
        for (int sample_index = 0; sample_index < samples; ++sample_index) {
            const double t = static_cast<double>(sample_index) / samples;
            double x, dx, ddx, y, dy, ddy, z, dz, ddz;
            eval_fourier_interleaved(coeffs_x + static_cast<size_t>(coil) * n_coeff, n_coeff, t, x, dx, ddx);
            eval_fourier_interleaved(coeffs_y + static_cast<size_t>(coil) * n_coeff, n_coeff, t, y, dy, ddy);
            eval_fourier_interleaved(coeffs_z + static_cast<size_t>(coil) * n_coeff, n_coeff, t, z, dz, ddz);
            GeometrySample item{{x, y, z}, {dx, dy, dz}, {ddx, ddy, ddz}, 0.0};
            const double speed = vec_norm(item.first);
            item.curvature = vec_norm(vec_cross(item.first, item.second)) /
                std::max(speed * speed * speed, 1.0e-30);
            const int flat = coil * samples + sample_index;
            geometry[flat] = item;
            curvatures[flat] = item.curvature;
            lengths[coil] += speed / samples;
        }
    }

    std::vector<int> curvature_order(sample_count);
    std::iota(curvature_order.begin(), curvature_order.end(), 0);
    std::sort(curvature_order.begin(), curvature_order.end(), [&](int left, int right) {
        return curvatures[left] < curvatures[right];
    });
    const double percentile_position = 0.95 * (sample_count - 1);
    const int percentile_lo = static_cast<int>(std::floor(percentile_position));
    const int percentile_hi = static_cast<int>(std::ceil(percentile_position));
    const double percentile_alpha = percentile_position - percentile_lo;
    std::vector<double> percentile_weights(sample_count, 0.0);
    percentile_weights[curvature_order[percentile_lo]] += 1.0 - percentile_alpha;
    percentile_weights[curvature_order[percentile_hi]] += percentile_alpha;
    const int maximum_curvature_index = static_cast<int>(
        std::max_element(curvatures.begin(), curvatures.end()) - curvatures.begin()
    );

    std::vector<FullGeometryPoint> full_points;
    full_points.reserve(static_cast<size_t>(n_base_coils) * 2 * nfp * samples);
    int label = 0;
    for (int coil = 0; coil < n_base_coils; ++coil) {
        for (int reflected = 0; reflected < 2; ++reflected) {
            for (int period = 0; period < nfp; ++period) {
                for (int sample_index = 0; sample_index < samples; ++sample_index) {
                    Vec3d point = geometry[coil * samples + sample_index].point;
                    if (reflected) {
                        point.y = -point.y;
                        point.z = -point.z;
                    }
                    full_points.push_back({
                        rotate_z(point, TWOPI * static_cast<double>(period) / nfp),
                        coil, sample_index, reflected, period, label,
                    });
                }
                ++label;
            }
        }
    }
    size_t spacing_left = 0;
    size_t spacing_right = 0;
    size_t radius_point = 0;
    double minimum_spacing = std::numeric_limits<double>::infinity();
    double minimum_radius = std::numeric_limits<double>::infinity();
    for (size_t left = 0; left < full_points.size(); ++left) {
        const Vec3d point = full_points[left].point;
        const double radius = std::hypot(point.x, point.y);
        if (radius < minimum_radius) {
            minimum_radius = radius;
            radius_point = left;
        }
        for (size_t right = left + 1; right < full_points.size(); ++right) {
            if (full_points[left].label == full_points[right].label) continue;
            const double distance = vec_norm(vec_sub(point, full_points[right].point));
            if (distance < minimum_spacing) {
                minimum_spacing = distance;
                spacing_left = left;
                spacing_right = right;
            }
        }
    }

    const CoilMetrics metrics = compute_coil_metrics(
        coeffs_x, coeffs_y, coeffs_z, currents_a, n_base_coils, n_coeff, nfp
    );
    output.value = 100.0 * coil_component(metrics);
    output.x.assign(parameter_count, 0.0);
    output.y.assign(parameter_count, 0.0);
    output.z.assign(parameter_count, 0.0);
    output.current.assign(n_base_coils, 0.0);

    std::vector<double> d_length[3];
    std::vector<double> d_p95[3];
    std::vector<double> d_max[3];
    std::vector<double> d_spacing[3];
    std::vector<double> d_radius[3];
    std::vector<double> d_high_mode[3];
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        d_length[coordinate].assign(parameter_count, 0.0);
        d_p95[coordinate].assign(parameter_count, 0.0);
        d_max[coordinate].assign(parameter_count, 0.0);
        d_spacing[coordinate].assign(parameter_count, 0.0);
        d_radius[coordinate].assign(parameter_count, 0.0);
        d_high_mode[coordinate].assign(parameter_count, 0.0);
    }
    for (int coil = 0; coil < n_base_coils; ++coil) {
        for (int coefficient = 0; coefficient < n_coeff; ++coefficient) {
            const int parameter = coil * n_coeff + coefficient;
            for (int sample_index = 0; sample_index < samples; ++sample_index) {
                const int flat = coil * samples + sample_index;
                const GeometrySample& item = geometry[flat];
                double basis, first, second;
                fourier_basis(n_coeff, static_cast<double>(sample_index) / samples, coefficient, basis, first, second);
                for (int coordinate = 0; coordinate < 3; ++coordinate) {
                    const Vec3d d_first = coordinate_vector(coordinate, first);
                    const Vec3d d_second = coordinate_vector(coordinate, second);
                    const double speed = vec_norm(item.first);
                    if (speed > 1.0e-15) {
                        d_length[coordinate][parameter] +=
                            vec_dot(item.first, d_first) /
                            (speed * samples * n_base_coils);
                    }
                    const double d_curvature = curvature_directional_derivative(
                        item, d_first, d_second
                    );
                    d_p95[coordinate][parameter] += percentile_weights[flat] * d_curvature;
                    if (flat == maximum_curvature_index) {
                        d_max[coordinate][parameter] = d_curvature;
                    }
                }
            }
        }
    }

    const FullGeometryPoint& spacing_a = full_points[spacing_left];
    const FullGeometryPoint& spacing_b = full_points[spacing_right];
    const Vec3d spacing_delta = vec_sub(spacing_a.point, spacing_b.point);
    const FullGeometryPoint& radial = full_points[radius_point];
    for (int coil = 0; coil < n_base_coils; ++coil) {
        for (int coefficient = 0; coefficient < n_coeff; ++coefficient) {
            const int parameter = coil * n_coeff + coefficient;
            double basis_a = 0.0, unused_first = 0.0, unused_second = 0.0;
            double basis_b = 0.0;
            double basis_r = 0.0;
            if (coil == spacing_a.coil) {
                fourier_basis(n_coeff, static_cast<double>(spacing_a.sample) / samples, coefficient, basis_a, unused_first, unused_second);
            }
            if (coil == spacing_b.coil) {
                fourier_basis(n_coeff, static_cast<double>(spacing_b.sample) / samples, coefficient, basis_b, unused_first, unused_second);
            }
            if (coil == radial.coil) {
                fourier_basis(n_coeff, static_cast<double>(radial.sample) / samples, coefficient, basis_r, unused_first, unused_second);
            }
            for (int coordinate = 0; coordinate < 3; ++coordinate) {
                Vec3d d_a{0.0, 0.0, 0.0};
                Vec3d d_b{0.0, 0.0, 0.0};
                if (coil == spacing_a.coil) {
                    d_a = transformed_basis_vector(
                        coordinate, basis_a, spacing_a.reflected, spacing_a.period, nfp
                    );
                }
                if (coil == spacing_b.coil) {
                    d_b = transformed_basis_vector(
                        coordinate, basis_b, spacing_b.reflected, spacing_b.period, nfp
                    );
                }
                if (minimum_spacing > 1.0e-15) {
                    d_spacing[coordinate][parameter] =
                        vec_dot(spacing_delta, vec_sub(d_a, d_b)) / minimum_spacing;
                }
                if (coil == radial.coil && minimum_radius > 1.0e-15) {
                    const Vec3d d_point = transformed_basis_vector(
                        coordinate, basis_r, radial.reflected, radial.period, nfp
                    );
                    d_radius[coordinate][parameter] =
                        (radial.point.x * d_point.x + radial.point.y * d_point.y) /
                        minimum_radius;
                }
            }
        }
    }

    const int order = (n_coeff - 1) / 2;
    const int high_start = std::max(1, static_cast<int>(std::floor(0.6 * order)));
    double total_energy = 0.0;
    double high_energy = 0.0;
    const double* blocks[] = {coeffs_x, coeffs_y, coeffs_z};
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        for (int coil = 0; coil < n_base_coils; ++coil) {
            const double* coefficients = blocks[coordinate] + static_cast<size_t>(coil) * n_coeff;
            for (int mode = 1; mode <= order; ++mode) {
                const double energy = coefficients[2 * mode - 1] * coefficients[2 * mode - 1] +
                    coefficients[2 * mode] * coefficients[2 * mode];
                total_energy += energy;
                if (mode >= high_start) high_energy += energy;
            }
        }
    }
    if (total_energy > 1.0e-30) {
        for (int coordinate = 0; coordinate < 3; ++coordinate) {
            for (int coil = 0; coil < n_base_coils; ++coil) {
                const double* coefficients = blocks[coordinate] + static_cast<size_t>(coil) * n_coeff;
                for (int coefficient = 1; coefficient < n_coeff; ++coefficient) {
                    const int mode = (coefficient + 1) / 2;
                    const double d_total = 2.0 * coefficients[coefficient];
                    const double d_high = mode >= high_start ? d_total : 0.0;
                    d_high_mode[coordinate][coil * n_coeff + coefficient] =
                        (d_high * total_energy - high_energy * d_total) /
                        (total_energy * total_energy);
                }
            }
        }
    }

    const double metric_derivatives[] = {
        0.16 * q_down_derivative(metrics.length_mean, 7.0, 1.4),
        0.20 * q_down_derivative(metrics.curvature_p95, 10.0, 1.3),
        0.12 * q_down_derivative(metrics.curvature_max, 35.0, 1.2),
        0.20 * q_up_derivative(metrics.min_intercoil_distance, 0.08, 1.1),
        0.12 * q_up_derivative(metrics.min_axis_distance, 0.20, 1.2),
        0.13 * q_down_derivative(metrics.high_mode_fraction, 0.05, 1.0),
    };
    std::vector<double>* outputs[] = {&output.x, &output.y, &output.z};
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        for (int parameter = 0; parameter < parameter_count; ++parameter) {
            (*outputs[coordinate])[parameter] = 100.0 * (
                metric_derivatives[0] * d_length[coordinate][parameter] +
                metric_derivatives[1] * d_p95[coordinate][parameter] +
                metric_derivatives[2] * d_max[coordinate][parameter] +
                metric_derivatives[3] * d_spacing[coordinate][parameter] +
                metric_derivatives[4] * d_radius[coordinate][parameter] +
                metric_derivatives[5] * d_high_mode[coordinate][parameter]
            );
        }
    }
    int active_current = 0;
    for (int coil = 1; coil < n_base_coils; ++coil) {
        if (std::abs(currents_a[coil]) > std::abs(currents_a[active_current])) active_current = coil;
    }
    if (currents_a[active_current] != 0.0) {
        output.current[active_current] = 100.0 * 0.07 *
            q_down_derivative(metrics.current_abs_max, 2.0e6, 1.0) *
            std::copysign(1.0, currents_a[active_current]);
    }
    return true;
}

struct FixedFrontG2Gradient {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
    std::vector<double> current;
    double point_vjp_s = 0.0;
    double field_vjp_s = 0.0;
    double parameter_map_s = 0.0;
};

struct QsErrorAdjoints {
    double target = 0.0;
    double target_edge = 0.0;
    double qa = 0.0;
    double qp = 0.0;
};

double helicity_quality_derivative(double advantage, const SgpuScoreConfig& config) {
    double derivative = 0.0;
    const double linear_position = advantage / config.score_qh_helicity_good;
    if (linear_position > 0.0 && linear_position < 1.0) {
        derivative += config.score_qh_helicity_exploration_fraction /
            config.score_qh_helicity_good;
    }
    const double window_position =
        (advantage - config.score_qh_helicity_bad) /
        (config.score_qh_helicity_good - config.score_qh_helicity_bad);
    if (window_position > 0.0 && window_position < 1.0) {
        derivative += (1.0 - config.score_qh_helicity_exploration_fraction) *
            6.0 * window_position * (1.0 - window_position) /
            (config.score_qh_helicity_good - config.score_qh_helicity_bad);
    }
    return derivative;
}

QsErrorAdjoints score_qs_error_adjoints(
    const SgpuScoreConfig& config,
    const SgpuScoreResult& result
) {
    QsErrorAdjoints output;
    double total_weight = 0.0;
    for (int component = 0; component < SGPU_SCORE_COMPONENT_COUNT; ++component) {
        total_weight += std::max(config.score_weights[component], 0.0);
    }
    if (!(total_weight > 0.0)) return output;
    const double d_global = q_down_derivative(
        result.qs_target_global_error_per_helicity, config.score_qs_global_scale, 0.9
    );
    const double d_edge = q_down_derivative(
        result.qs_target_edge_error_per_helicity, config.score_qs_edge_scale, 0.9
    );
    const double volume_scale = result.score_volume_qs_size_factor *
        result.score_volume_qs_iota_factor;
    const double score_before_scale =
        100.0 * std::max(config.score_weights[SGPU_SCORE_COMPONENT_VOLUME_QS], 0.0) /
        total_weight * volume_scale;
    const double d_before_target = score_before_scale * 0.80 * d_global;
    const double d_before_edge = score_before_scale * 0.20 * d_edge;
    const double iota_gate = result.score_qh_total_iota_factor;
    const double helicity_gate = result.score_qh_total_helicity_factor;
    output.target = iota_gate * helicity_gate * d_before_target;
    output.target_edge = iota_gate * helicity_gate * d_before_edge;

    const bool qh_target = config.target_M != 0 && config.target_N != 0;
    if (!qh_target) return output;
    const double target_error = result.qs_target_global_error_per_helicity;
    const bool qa_competitor =
        result.qs_qa_global_error_per_helicity <= result.qs_qp_global_error_per_helicity;
    const double competitor = qa_competitor
        ? result.qs_qa_global_error_per_helicity
        : result.qs_qp_global_error_per_helicity;
    const double denominator = target_error + competitor;
    if (!(denominator > 1.0e-30)) return output;
    const double raw_advantage = competitor / denominator;
    if (!(raw_advantage > 0.0 && raw_advantage < 1.0)) return output;
    const double d_quality = helicity_quality_derivative(raw_advantage, config);
    const double d_gate_d_advantage =
        (1.0 - config.score_qh_total_helicity_floor) * d_quality;
    const double common = iota_gate * result.score_before_qh_iota_gate *
        d_gate_d_advantage;
    output.target += common * (-competitor / (denominator * denominator));
    const double d_competitor = common * target_error / (denominator * denominator);
    if (qa_competitor) {
        output.qa = d_competitor;
    } else {
        output.qp = d_competitor;
    }
    return output;
}

bool map_fixed_front_field_adjoint(
    FixedFrontG2Cache& cache,
    const DeviceBuffer<float>& d_adj_B,
    const DeviceBuffer<float>& d_adj_grad_B,
    const DeviceBuffer<double>& d_adj_G,
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig& config,
    SgpuScoreResult& result,
    FixedFrontG2Gradient& output
) {
    const int segment_count = sgpu_segment_count(cache.field);
    DeviceBuffer<float> d_adj_segment_position(static_cast<size_t>(segment_count) * 3);
    DeviceBuffer<float> d_adj_segment_weight(static_cast<size_t>(segment_count) * 3);
    if (segment_count <= 0 || !d_adj_segment_position.data() || !d_adj_segment_weight.data()) {
        fail_result(&result, "fixed-front segment-adjoint allocation failed");
        return false;
    }
    auto started = Clock::now();
    if (sgpu_internal_B_grad_segment_vjp_f32_device(
            cache.field, cache.points.xyz.data(), d_adj_B.data(), d_adj_grad_B.data(),
            cache.points.count, d_adj_segment_position.data(), d_adj_segment_weight.data()) ||
        !cuda_stage_ok(cudaDeviceSynchronize(), result, "fixed-front Biot-Savart VJP")) {
        if (!result.error_message[0]) fail_from_backend(&result, "fixed-front Biot-Savart VJP");
        return false;
    }
    output.field_vjp_s = seconds_since(started);

    started = Clock::now();
    std::vector<float> adj_segment_position(static_cast<size_t>(segment_count) * 3);
    std::vector<float> adj_segment_weight(static_cast<size_t>(segment_count) * 3);
    double adj_G = 0.0;
    if (cudaMemcpy(
            adj_segment_position.data(), d_adj_segment_position.data(),
            adj_segment_position.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(
            adj_segment_weight.data(), d_adj_segment_weight.data(),
            adj_segment_weight.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(&adj_G, d_adj_G.data(), sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fail_result(&result, "fixed-front segment-adjoint copy failed");
        return false;
    }
    const int parameter_count = n_base_coils * n_coeff;
    output.x.assign(parameter_count, 0.0);
    output.y.assign(parameter_count, 0.0);
    output.z.assign(parameter_count, 0.0);
    output.current.assign(n_base_coils, 0.0);
    std::vector<double>* coordinate_outputs[] = {&output.x, &output.y, &output.z};
    int segment = 0;
    for (int coil = 0; coil < n_base_coils; ++coil) {
        for (int sample = 0; sample < config.segments_per_coil; ++sample) {
            const double t = (static_cast<double>(sample) + 0.5) / config.segments_per_coil;
            double px, vx, unused, py, vy, pz, vz;
            eval_fourier_interleaved(
                coeffs_x + static_cast<size_t>(coil) * n_coeff, n_coeff, t, px, vx, unused
            );
            eval_fourier_interleaved(
                coeffs_y + static_cast<size_t>(coil) * n_coeff, n_coeff, t, py, vy, unused
            );
            eval_fourier_interleaved(
                coeffs_z + static_cast<size_t>(coil) * n_coeff, n_coeff, t, pz, vz, unused
            );
            const Vec3d base_velocity{
                vx / config.segments_per_coil,
                vy / config.segments_per_coil,
                vz / config.segments_per_coil,
            };
            for (int period = 0; period < nfp; ++period) {
                for (int reflected = 0; reflected < 2; ++reflected) {
                    if (segment >= segment_count) {
                        fail_result(&result, "fixed-front segment ordering mismatch");
                        return false;
                    }
                    const Vec3d adj_position{
                        adj_segment_position[3 * segment],
                        adj_segment_position[3 * segment + 1],
                        adj_segment_position[3 * segment + 2],
                    };
                    const Vec3d adj_weight{
                        adj_segment_weight[3 * segment],
                        adj_segment_weight[3 * segment + 1],
                        adj_segment_weight[3 * segment + 2],
                    };
                    const double current_sign = reflected ? -1.0 : 1.0;
                    for (int coordinate = 0; coordinate < 3; ++coordinate) {
                        for (int coefficient = 0; coefficient < n_coeff; ++coefficient) {
                            double basis, first, second;
                            fourier_basis(n_coeff, t, coefficient, basis, first, second);
                            const Vec3d d_position = transformed_basis_vector(
                                coordinate, basis, reflected, period, nfp
                            );
                            Vec3d d_weight = transformed_basis_vector(
                                coordinate,
                                first / config.segments_per_coil,
                                reflected,
                                period,
                                nfp
                            );
                            d_weight.x *= current_sign * currents_a[coil];
                            d_weight.y *= current_sign * currents_a[coil];
                            d_weight.z *= current_sign * currents_a[coil];
                            (*coordinate_outputs[coordinate])[coil * n_coeff + coefficient] +=
                                vec_dot(adj_position, d_position) + vec_dot(adj_weight, d_weight);
                        }
                    }
                    Vec3d d_weight_current = base_velocity;
                    if (reflected) {
                        d_weight_current.y = -d_weight_current.y;
                        d_weight_current.z = -d_weight_current.z;
                    }
                    d_weight_current = rotate_z(
                        d_weight_current, TWOPI * static_cast<double>(period) / nfp
                    );
                    d_weight_current.x *= current_sign;
                    d_weight_current.y *= current_sign;
                    d_weight_current.z *= current_sign;
                    output.current[coil] += vec_dot(adj_weight, d_weight_current);
                    ++segment;
                }
            }
        }
    }
    if (segment != segment_count) {
        fail_result(&result, "fixed-front segment count mismatch");
        return false;
    }
    const double G_current_scale = std::copysign(4.0e-7 * nfp, cache.G);
    for (int coil = 0; coil < n_base_coils; ++coil) {
        if (currents_a[coil] != 0.0) {
            output.current[coil] +=
                adj_G * G_current_scale * std::copysign(1.0, currents_a[coil]);
        }
    }
    output.parameter_map_s = seconds_since(started);
    return true;
}

bool compute_fixed_front_g2_gradient(
    FixedFrontG2Cache& cache,
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig& config,
    SgpuScoreResult& result,
    FixedFrontG2Gradient& output
) {
    if (!cache.ready || !cache.field || cache.points.count <= 0) {
        fail_result(&result, "fixed-front G2 cache is incomplete");
        return false;
    }
    const int point_count = cache.points.count;
    DeviceBuffer<float> d_iota;
    DeviceBuffer<float> d_adj_B(static_cast<size_t>(point_count) * 3);
    DeviceBuffer<float> d_adj_grad_B(static_cast<size_t>(point_count) * 9);
    DeviceBuffer<double> d_adj_G(1);
    if (!copy_to_device(d_iota, cache.alpha.iota_coefficients) ||
        !d_adj_B.data() || !d_adj_grad_B.data() || !d_adj_G.data() ||
        cudaMemset(d_adj_G.data(), 0, sizeof(double)) != cudaSuccess) {
        fail_result(&result, "fixed-front G2 point-adjoint allocation failed");
        return false;
    }
    const QsErrorAdjoints errors = score_qs_error_adjoints(config, result);
    const double target_norm = std::max(
        std::hypot(static_cast<double>(config.target_M), static_cast<double>(config.target_N)),
        1.0
    );
    const double qp_norm = std::max(std::abs(nfp), 1);
    const float edge_threshold = static_cast<float>(
        config.volume_rho_min + (1.0 - config.volume_rho_min) *
        (config.radial_bin_count - 1.0) / config.radial_bin_count
    );
    auto started = Clock::now();
    constexpr int threads = 256;
    compute_qs_point_adjoint_kernel<<<(point_count + threads - 1) / threads, threads>>>(
        cache.B.data(), cache.grad_B.data(), cache.points.grad_s.data(),
        cache.points.flux_derivative.data(), cache.points.rho.data(),
        cache.points.volume_weight.data(), point_count, d_iota.data(), config.iota_degree,
        config.target_M, config.target_N, nfp, cache.G, edge_threshold,
        cache.qs_sums[0], cache.qs_sums[2], result.qs_global_error,
        result.qs_edge_error, result.qs_qa_global_error,
        result.qs_qp_global_error_per_helicity * qp_norm,
        errors.target, errors.target_edge, errors.qa, errors.qp,
        d_adj_B.data(), d_adj_grad_B.data(), d_adj_G.data()
    );
    if (!cuda_stage_ok(cudaDeviceSynchronize(), result, "fixed-front QS point VJP")) return false;
    output.point_vjp_s = seconds_since(started);
    return map_fixed_front_field_adjoint(
        cache, d_adj_B, d_adj_grad_B, d_adj_G,
        coeffs_x, coeffs_y, coeffs_z, currents_a,
        n_base_coils, n_coeff, nfp, config, result, output
    );
}

bool evaluate_fixed_front_g2_scalar(
    const FixedFrontG2Cache& cache,
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig& config,
    const SgpuScoreResult& center,
    double& frozen_score,
    double& volume_component,
    double& coil_score,
    double& target_error,
    double& qa_error,
    double& qp_error,
    std::string& error
) {
    void* field = nullptr;
    if (sgpu_create_field(
            coeffs_x, coeffs_y, coeffs_z, currents_a, n_base_coils, n_coeff,
            nfp, config.segments_per_coil, config.device_id, &field)) {
        error = "fixed-front query field creation failed";
        return false;
    }
    DeviceBuffer<float> d_B(static_cast<size_t>(cache.points.count) * 3);
    DeviceBuffer<float> d_grad_B(static_cast<size_t>(cache.points.count) * 9);
    SgpuScoreResult query;
    initialize_result(&query, config.device_id);
    bool ok = d_B.data() && d_grad_B.data();
    if (!ok) {
        error = "fixed-front query field allocation failed";
    } else if (sgpu_internal_eval_B_grad_f32_device(
                   field, cache.points.xyz.data(), d_B.data(), d_grad_B.data(),
                   cache.points.count) ||
               !cuda_stage_ok(cudaDeviceSynchronize(), query, "fixed-front query B/grad(B)")) {
        error = query.error_message[0]
            ? query.error_message : "fixed-front query B/grad(B) failed";
        ok = false;
    } else if (!compute_qs_metric_native(
                   cache.points, d_B.data(), d_grad_B.data(), cache.alpha,
                   currents_a, n_base_coils, nfp, center.flux_edge, config,
                   query, nullptr)) {
        error = query.error_message[0]
            ? query.error_message : "fixed-front query QS metric failed";
        ok = false;
    }
    sgpu_destroy_field(field);
    if (!ok) return false;

    const CoilMetrics metrics = compute_coil_metrics(
        coeffs_x, coeffs_y, coeffs_z, currents_a,
        n_base_coils, n_coeff, nfp
    );
    const double coil_unit = coil_component(metrics);
    const double global_score = q_down(
        query.qs_target_global_error_per_helicity,
        config.score_qs_global_scale,
        0.9
    );
    const double edge_score = q_down(
        query.qs_target_edge_error_per_helicity,
        config.score_qs_edge_scale,
        0.9,
        global_score
    );
    const double residual_score = blend({{0.80, global_score}, {0.20, edge_score}});
    const double volume_unit = residual_score *
        center.score_volume_qs_size_factor * center.score_volume_qs_iota_factor;
    const bool qh_target = config.target_M != 0 && config.target_N != 0;
    const double competitor_error = std::min(
        query.qs_qa_global_error_per_helicity,
        query.qs_qp_global_error_per_helicity
    );
    const double advantage = !qh_target ? 1.0 : competitor_error / std::max(
        query.qs_target_global_error_per_helicity + competitor_error,
        1.0e-300
    );
    const double helicity_position = clip01(
        (advantage - config.score_qh_helicity_bad) /
        (config.score_qh_helicity_good - config.score_qh_helicity_bad)
    );
    const double helicity_linear = clip01(
        advantage / config.score_qh_helicity_good
    );
    const double helicity_window = helicity_position * helicity_position *
        (3.0 - 2.0 * helicity_position);
    const double helicity_quality = !qh_target ? 1.0 :
        config.score_qh_helicity_exploration_fraction * helicity_linear +
        (1.0 - config.score_qh_helicity_exploration_fraction) * helicity_window;
    const double helicity_factor = !qh_target ? 1.0 :
        config.score_qh_total_helicity_floor +
        (1.0 - config.score_qh_total_helicity_floor) * helicity_quality;

    double total_weight = 0.0;
    double weighted_score = 0.0;
    for (int component = 0; component < SGPU_SCORE_COMPONENT_COUNT; ++component) {
        const double weight = std::max(config.score_weights[component], 0.0);
        double value = center.components[component] / 100.0;
        if (component == SGPU_SCORE_COMPONENT_VOLUME_QS) value = volume_unit;
        if (component == SGPU_SCORE_COMPONENT_COIL) value = coil_unit;
        total_weight += weight;
        weighted_score += weight * clip01(value);
    }
    const double score_before = total_weight > 0.0
        ? 100.0 * clip01(weighted_score / total_weight) : 0.0;
    frozen_score = score_before * center.score_qh_total_iota_factor * helicity_factor;
    volume_component = 100.0 * clip01(volume_unit);
    coil_score = 100.0 * clip01(coil_unit);
    target_error = query.qs_target_global_error_per_helicity;
    qa_error = query.qs_qa_global_error_per_helicity;
    qp_error = query.qs_qp_global_error_per_helicity;
    return true;
}

bool evaluate_fixed_front_g3_scalar(
    const FixedFrontG2Cache& cache,
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig& config,
    const SgpuScoreResult& center,
    double& frozen_score,
    double& volume_component,
    double& coordinate_component,
    double& iota_component,
    double& coil_score,
    double& target_error,
    double& qa_error,
    double& qp_error,
    double& iota_minimum,
    double& iota_maximum,
    std::string& error
) {
    void* field = nullptr;
    if (sgpu_create_field(
            coeffs_x, coeffs_y, coeffs_z, currents_a, n_base_coils, n_coeff,
            nfp, config.segments_per_coil, config.device_id, &field)) {
        error = "fixed-front G3 query field creation failed";
        return false;
    }
    DeviceBuffer<float> d_B(static_cast<size_t>(cache.points.count) * 3);
    DeviceBuffer<float> d_grad_B(static_cast<size_t>(cache.points.count) * 9);
    SgpuScoreResult query;
    initialize_result(&query, config.device_id);
    AlphaFitNative alpha;
    bool ok = d_B.data() && d_grad_B.data();
    if (!ok) {
        error = "fixed-front G3 query field allocation failed";
    } else if (sgpu_internal_eval_B_grad_f32_device(
                   field, cache.points.xyz.data(), d_B.data(), d_grad_B.data(),
                   cache.points.count) ||
               !cuda_stage_ok(cudaDeviceSynchronize(), query, "fixed-front G3 query B/grad(B)")) {
        error = query.error_message[0]
            ? query.error_message : "fixed-front G3 query B/grad(B) failed";
        ok = false;
    } else if (!fit_alpha_native(cache.points, d_B.data(), nfp, config, alpha, query)) {
        error = query.error_message[0]
            ? query.error_message : "fixed-front G3 query alpha/iota fit failed";
        ok = false;
    } else if (!compute_qs_metric_native(
                   cache.points, d_B.data(), d_grad_B.data(), alpha,
                   currents_a, n_base_coils, nfp, center.flux_edge, config,
                   query, nullptr)) {
        error = query.error_message[0]
            ? query.error_message : "fixed-front G3 query QS metric failed";
        ok = false;
    }
    sgpu_destroy_field(field);
    if (!ok) return false;

    iota_minimum = std::numeric_limits<double>::infinity();
    iota_maximum = -std::numeric_limits<double>::infinity();
    const double u_min = config.volume_rho_min * config.volume_rho_min;
    for (int sample = 0; sample <= 256; ++sample) {
        const double u = u_min + (1.0 - u_min) * sample / 256.0;
        double iota = 0.0;
        double power = 1.0;
        for (float coefficient : alpha.iota_coefficients) {
            iota += coefficient * power;
            power *= u;
        }
        iota_minimum = std::min(iota_minimum, iota);
        iota_maximum = std::max(iota_maximum, iota);
    }

    const bool qh_target = config.target_M != 0 && config.target_N != 0;
    const double iota_unit = !qh_target ? 1.0 : std::pow(
        clip01(minimum_absolute_iota(iota_minimum, iota_maximum) /
               config.score_qh_iota_threshold),
        config.score_qh_iota_power
    );
    const double iota_factor = !qh_target ? 1.0 :
        config.score_volume_qs_iota_floor +
        (1.0 - config.score_volume_qs_iota_floor) * iota_unit;
    const double total_iota_factor = !qh_target ? 1.0 :
        config.score_qh_total_iota_floor +
        (1.0 - config.score_qh_total_iota_floor) * iota_unit;
    const double global_score = q_down(
        query.qs_target_global_error_per_helicity,
        config.score_qs_global_scale,
        0.9
    );
    const double edge_score = q_down(
        query.qs_target_edge_error_per_helicity,
        config.score_qs_edge_scale,
        0.9,
        global_score
    );
    const double residual_score = blend({{0.80, global_score}, {0.20, edge_score}});
    const double volume_unit = residual_score *
        center.score_volume_qs_size_factor * iota_factor;
    const double alpha_score = q_down(
        alpha.relative_l2, config.score_alpha_relative_l2_scale, 1.0
    );
    const double center_alpha_score = q_down(
        center.alpha_relative_l2, config.score_alpha_relative_l2_scale, 1.0
    );
    const double coordinate_unit = center.components[SGPU_SCORE_COMPONENT_COORDINATE] / 100.0 +
        0.20 * (alpha_score - center_alpha_score);
    const CoilMetrics metrics = compute_coil_metrics(
        coeffs_x, coeffs_y, coeffs_z, currents_a,
        n_base_coils, n_coeff, nfp
    );
    const double coil_unit = coil_component(metrics);
    const double competitor_error = std::min(
        query.qs_qa_global_error_per_helicity,
        query.qs_qp_global_error_per_helicity
    );
    const double advantage = !qh_target ? 1.0 : competitor_error / std::max(
        query.qs_target_global_error_per_helicity + competitor_error,
        1.0e-300
    );
    const double helicity_position = clip01(
        (advantage - config.score_qh_helicity_bad) /
        (config.score_qh_helicity_good - config.score_qh_helicity_bad)
    );
    const double helicity_linear = clip01(
        advantage / config.score_qh_helicity_good
    );
    const double helicity_window = helicity_position * helicity_position *
        (3.0 - 2.0 * helicity_position);
    const double helicity_quality = !qh_target ? 1.0 :
        config.score_qh_helicity_exploration_fraction * helicity_linear +
        (1.0 - config.score_qh_helicity_exploration_fraction) * helicity_window;
    const double helicity_factor = !qh_target ? 1.0 :
        config.score_qh_total_helicity_floor +
        (1.0 - config.score_qh_total_helicity_floor) * helicity_quality;

    double total_weight = 0.0;
    double weighted_score = 0.0;
    for (int component = 0; component < SGPU_SCORE_COMPONENT_COUNT; ++component) {
        const double weight = std::max(config.score_weights[component], 0.0);
        double value = center.components[component] / 100.0;
        if (component == SGPU_SCORE_COMPONENT_COORDINATE) value = coordinate_unit;
        if (component == SGPU_SCORE_COMPONENT_VOLUME_QS) value = volume_unit;
        if (component == SGPU_SCORE_COMPONENT_IOTA) value = iota_unit;
        if (component == SGPU_SCORE_COMPONENT_COIL) value = coil_unit;
        total_weight += weight;
        weighted_score += weight * clip01(value);
    }
    const double score_before = total_weight > 0.0
        ? 100.0 * clip01(weighted_score / total_weight) : 0.0;
    frozen_score = score_before * total_iota_factor * helicity_factor;
    volume_component = 100.0 * clip01(volume_unit);
    coordinate_component = 100.0 * clip01(coordinate_unit);
    iota_component = 100.0 * clip01(iota_unit);
    coil_score = 100.0 * clip01(coil_unit);
    target_error = query.qs_target_global_error_per_helicity;
    qa_error = query.qs_qa_global_error_per_helicity;
    qp_error = query.qs_qp_global_error_per_helicity;
    return true;
}

bool evaluate_fixed_branch_g4_scalar(
    const FixedFrontG2Cache& cache,
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig& config,
    const SgpuScoreResult& center,
    SgpuScoreResult& query,
    std::string& error
) {
    initialize_result(&query, config.device_id);
    const auto total_started = Clock::now();
    const CoilMetrics coil = compute_coil_metrics(
        coeffs_x, coeffs_y, coeffs_z, currents_a,
        n_base_coils, n_coeff, nfp
    );
    query.coil_length_mean = coil.length_mean;
    query.coil_curvature_p95 = coil.curvature_p95;
    query.coil_curvature_max = coil.curvature_max;
    query.coil_min_intercoil_distance = coil.min_intercoil_distance;
    query.coil_min_axis_distance = coil.min_axis_distance;
    query.coil_high_mode_energy_fraction = coil.high_mode_fraction;
    query.coil_current_abs_max_a = coil.current_abs_max;
    query.axis_R = center.axis_R;
    query.axis_Z = center.axis_Z;
    query.axis_residual = center.axis_residual;
    query.axis_topology_trace = center.axis_topology_trace;
    query.axis_topology_det = center.axis_topology_det;
    query.axis_ellipse_aspect = center.axis_ellipse_aspect;
    query.axis_candidate_count = center.axis_candidate_count;
    query.stable_surface_count = center.stable_surface_count;
    query.stage_completed = SCORE_STAGE_AXIS;

    void* field = nullptr;
    auto started = Clock::now();
    if (sgpu_create_field(
            coeffs_x, coeffs_y, coeffs_z, currents_a, n_base_coils, n_coeff,
            nfp, config.segments_per_coil, config.device_id, &field)) {
        error = "fixed-branch G4 query field creation failed";
        return false;
    }
    query.timings[SGPU_SCORE_TIME_FIELD_CREATE] = seconds_since(started);
    query.stage_completed = SCORE_STAGE_FIELD;

    PsiData psi;
    bool ok = fit_psi_native(
        field, cache.axis, nfp, config, psi,
        query.timings[SGPU_SCORE_TIME_PSI_POINTS],
        query.timings[SGPU_SCORE_TIME_PSI_FIT]
    );
    if (!ok) {
        error = "fixed-branch G4 query psi fit failed";
    } else {
        started = Clock::now();
        ok = validate_psi_native(field, cache.axis, nfp, config, psi);
        query.timings[SGPU_SCORE_TIME_PSI_VALIDATE] = seconds_since(started);
        if (!ok) error = "fixed-branch G4 query psi validation failed";
    }
    if (ok) {
        query.psi_train_rms = psi.train_rms;
        query.psi_angle_mean = psi.angle_mean;
        query.psi_angle_p95 = psi.angle_p95;
        query.psi_angle_l2 = psi.angle_l2;
        query.stage_completed = SCORE_STAGE_PSI;

        SurfaceScreen selected;
        selected.level = center.surface_level;
        selected.relative_drift_p95 = center.surface_drift_relative_p95;
        selected.one_period_relative_drift_p95 = center.surface_one_period_drift_relative_p95;
        selected.long_trace_periods_completed = center.surface_long_trace_periods_completed;
        selected.stable = true;
        selected.strict = true;
        selected.verified = true;
        selected.long_verified = true;
        std::vector<SurfaceScreen> screens{selected};
        ok = run_downstream_gpu(
            field, currents_a, n_base_coils, nfp, cache.axis, psi,
            screens, config, query, false
        );
        fill_early_components(config, coil, cache.axis, psi, screens, query);

        // Axis, drift and discrete surface-count terms define the fixed branch.
        // Only the continuously recomputed flux volume changes the surface-size term.
        query.components[SGPU_SCORE_COMPONENT_AXIS] =
            center.components[SGPU_SCORE_COMPONENT_AXIS] / 100.0;
        query.components[SGPU_SCORE_COMPONENT_SURFACE] =
            center.components[SGPU_SCORE_COMPONENT_SURFACE] / 100.0 +
            0.65 * (query.score_surface_size - center.score_surface_size);
        if (!ok && query.status == SGPU_SCORE_INTERNAL_ERROR) {
            error = query.error_message[0]
                ? query.error_message : "fixed-branch G4 downstream query failed";
        }
    }
    sgpu_destroy_field(field);
    if (!ok && query.status == SGPU_SCORE_INTERNAL_ERROR) return false;

    if (query.status == SGPU_SCORE_OK &&
        center.volume_candidate_count == center.volume_available_count &&
        query.volume_candidate_count != query.volume_available_count) {
        query.status = SGPU_SCORE_FLUX_REJECTED;
        std::snprintf(
            query.error_message,
            sizeof(query.error_message),
            "%s",
            "fixed-branch G4 volume active set changed"
        );
    }
    started = Clock::now();
    finalize_score(config, query);
    query.timings[SGPU_SCORE_TIME_SCORE] = seconds_since(started);
    query.timings[SGPU_SCORE_TIME_TOTAL] = seconds_since(total_started);
    return true;
}

void add_iota_score_solution_adjoint(
    const AlphaFitNative& alpha,
    const SgpuScoreConfig& config,
    const SgpuScoreResult& result,
    std::vector<double>& adj_solution
) {
    if (config.target_M == 0 || config.target_N == 0 || alpha.iota_coefficients.empty()) return;
    double iota_min = std::numeric_limits<double>::infinity();
    double iota_max = -std::numeric_limits<double>::infinity();
    double u_at_min = 0.0;
    double u_at_max = 0.0;
    const double u_min = config.volume_rho_min * config.volume_rho_min;
    for (int sample = 0; sample <= 256; ++sample) {
        const double u = u_min + (1.0 - u_min) * sample / 256.0;
        double value = 0.0;
        double power = 1.0;
        for (float coefficient : alpha.iota_coefficients) {
            value += coefficient * power;
            power *= u;
        }
        if (value < iota_min) {
            iota_min = value;
            u_at_min = u;
        }
        if (value > iota_max) {
            iota_max = value;
            u_at_max = u;
        }
    }
    if (iota_min <= 0.0 && iota_max >= 0.0) return;
    const bool use_min = std::abs(iota_min) <= std::abs(iota_max);
    const double active_iota = use_min ? iota_min : iota_max;
    const double active_u = use_min ? u_at_min : u_at_max;
    const double magnitude = std::abs(active_iota);
    if (!(magnitude > 0.0) || magnitude >= config.score_qh_iota_threshold) return;
    const double ratio = magnitude / config.score_qh_iota_threshold;
    const double d_iota_score_d_magnitude = config.score_qh_iota_power *
        std::pow(ratio, config.score_qh_iota_power - 1.0) /
        config.score_qh_iota_threshold;
    double total_weight = 0.0;
    for (int component = 0; component < SGPU_SCORE_COMPONENT_COUNT; ++component) {
        total_weight += std::max(config.score_weights[component], 0.0);
    }
    if (!(total_weight > 0.0)) return;
    const double iota_component_active =
        result.components[SGPU_SCORE_COMPONENT_IOTA] > 0.0 &&
        result.components[SGPU_SCORE_COMPONENT_IOTA] < 100.0;
    const double volume_component_active =
        result.components[SGPU_SCORE_COMPONENT_VOLUME_QS] > 0.0 &&
        result.components[SGPU_SCORE_COMPONENT_VOLUME_QS] < 100.0;
    const double d_before = 100.0 / total_weight * (
        (iota_component_active ? std::max(config.score_weights[SGPU_SCORE_COMPONENT_IOTA], 0.0) : 0.0) +
        (volume_component_active
            ? std::max(config.score_weights[SGPU_SCORE_COMPONENT_VOLUME_QS], 0.0) *
                result.score_qs_residual * result.score_volume_qs_size_factor *
                (1.0 - config.score_volume_qs_iota_floor)
            : 0.0)
    );
    const double d_score_d_iota_score = result.score_qh_total_helicity_factor * (
        result.score_qh_total_iota_factor * d_before +
        result.score_before_qh_iota_gate * (1.0 - config.score_qh_total_iota_floor)
    );
    const double active_scale = d_score_d_iota_score * d_iota_score_d_magnitude *
        std::copysign(1.0, active_iota);
    double power = 1.0;
    for (size_t degree = 0; degree < alpha.iota_coefficients.size(); ++degree) {
        adj_solution[alpha.mode_count + degree] += active_scale * power;
        power *= active_u;
    }
}

bool compute_alpha_iota_g3_gradient(
    FixedFrontG2Cache& cache,
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig& config,
    SgpuScoreResult& result,
    FixedFrontG2Gradient& output
) {
    AlphaFitNative& alpha = cache.alpha;
    if (!cache.ready || !cache.field || !alpha.adjoint_ready || config.alpha_solver_mode != 2 ||
        alpha.fit_count <= 0 || alpha.column_count <= 0 || alpha.qr_rows <= 0) {
        fail_result(&result, "G3 alpha QR adjoint cache is incomplete");
        return false;
    }
    const auto started = Clock::now();
    constexpr int threads = 256;
    DeviceBuffer<float> d_iota;
    DeviceBuffer<double> d_adj_iota(config.iota_degree + 1);
    if (!copy_to_device(d_iota, alpha.iota_coefficients) || !d_adj_iota.data() ||
        cudaMemset(d_adj_iota.data(), 0, d_adj_iota.size() * sizeof(double)) != cudaSuccess) {
        fail_result(&result, "G3 iota adjoint allocation failed");
        return false;
    }
    const QsErrorAdjoints errors = score_qs_error_adjoints(config, result);
    const float edge_threshold = static_cast<float>(
        config.volume_rho_min + (1.0 - config.volume_rho_min) *
        (config.radial_bin_count - 1.0) / config.radial_bin_count
    );
    compute_qs_iota_adjoint_kernel<<<
        (cache.points.count + threads - 1) / threads, threads
    >>>(
        cache.B.data(), cache.grad_B.data(), cache.points.grad_s.data(),
        cache.points.flux_derivative.data(), cache.points.rho.data(),
        cache.points.volume_weight.data(), cache.points.count, d_iota.data(),
        config.iota_degree, config.target_M, config.target_N, cache.G, edge_threshold,
        cache.qs_sums[0], cache.qs_sums[2], result.qs_global_error,
        result.qs_edge_error, result.qs_qa_global_error,
        errors.target, errors.target_edge, errors.qa, d_adj_iota.data()
    );
    std::vector<double> adj_iota(config.iota_degree + 1, 0.0);
    if (!cuda_stage_ok(cudaMemcpy(
            adj_iota.data(), d_adj_iota.data(),
            adj_iota.size() * sizeof(double), cudaMemcpyDeviceToHost
        ), result, "G3 iota adjoint copy")) {
        return false;
    }
    std::vector<double> adj_solution(alpha.column_count, 0.0);
    for (size_t degree = 0; degree < adj_iota.size(); ++degree) {
        adj_solution[alpha.mode_count + degree] = adj_iota[degree];
    }
    add_iota_score_solution_adjoint(alpha, config, result, adj_solution);

    double total_weight = 0.0;
    for (int component = 0; component < SGPU_SCORE_COMPONENT_COUNT; ++component) {
        total_weight += std::max(config.score_weights[component], 0.0);
    }
    const bool coordinate_active = result.components[SGPU_SCORE_COMPONENT_COORDINATE] > 0.0 &&
        result.components[SGPU_SCORE_COMPONENT_COORDINATE] < 100.0;
    const double coordinate_scale = coordinate_active && total_weight > 0.0
        ? result.score_qh_total_iota_factor * result.score_qh_total_helicity_factor *
            100.0 * std::max(config.score_weights[SGPU_SCORE_COMPONENT_COORDINATE], 0.0) /
            total_weight
        : 0.0;
    const float adj_relative = static_cast<float>(
        coordinate_scale * 0.20 * q_down_derivative(
            result.alpha_relative_l2, config.score_alpha_relative_l2_scale, 1.0
        )
    );
    // The normal-field residual is covariant with the fitted psi geometry. Its
    // frozen-surface partial is singular near B.normal = 0 and must be added
    // together with the compensating psi/surface-motion terms in G4.
    constexpr float adj_normal = 0.0f;
    std::vector<float> adj_solution_f(alpha.column_count);
    std::transform(adj_solution.begin(), adj_solution.end(), adj_solution_f.begin(), [](double value) {
        return static_cast<float>(value);
    });
    DeviceBuffer<float> d_adj_solution;
    DeviceBuffer<float> d_adj_residual(alpha.fit_count), d_adj_rhs(alpha.fit_count);
    DeviceBuffer<float> d_adj_scaled_solution(alpha.column_count);
    DeviceBuffer<float> d_multiplier(alpha.column_count);
    DeviceBuffer<float> d_matrix_multiplier(alpha.fit_count);
    DeviceBuffer<float> d_dot_adjoint_matrix(alpha.column_count);
    DeviceBuffer<float> d_adj_weight(alpha.fit_count), d_adj_b_theta(alpha.fit_count), d_adj_b_phi(alpha.fit_count);
    DeviceBuffer<int> d_mode_m, d_mode_n, d_mode_kind;
    DeviceBuffer<float> d_radial_coefficients;
    if (!copy_to_device(d_adj_solution, adj_solution_f) ||
        !copy_to_device(d_mode_m, alpha.modes.m) || !copy_to_device(d_mode_n, alpha.modes.n) ||
        !copy_to_device(d_mode_kind, alpha.modes.kind) ||
        !copy_to_device(d_radial_coefficients, alpha.modes.radial_coefficients) ||
        !d_adj_residual.data() || !d_adj_rhs.data() || !d_adj_scaled_solution.data() ||
        !d_multiplier.data() || !d_matrix_multiplier.data() || !d_dot_adjoint_matrix.data() ||
        !d_adj_weight.data() || !d_adj_b_theta.data() || !d_adj_b_phi.data()) {
        fail_result(&result, "G3 alpha LS adjoint allocation failed");
        return false;
    }
    alpha_residual_adjoint_kernel<<<(alpha.fit_count + threads - 1) / threads, threads>>>(
        alpha.residual.data(), alpha.rhs_reference.data(), alpha.fit_count,
        static_cast<float>(alpha.residual_norm), static_cast<float>(alpha.rhs_norm),
        adj_relative, d_adj_residual.data(), d_adj_rhs.data()
    );
    cublasHandle_t blas = nullptr;
    if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) {
        fail_result(&result, "G3 cuBLAS initialization failed");
        return false;
    }
    const float one = 1.0f;
    const float zero = 0.0f;
    cublasStatus_t status = cublasSgemv(
        blas, CUBLAS_OP_T, alpha.fit_count, alpha.column_count,
        &one, alpha.matrix_reference.data(), alpha.fit_count,
        d_adj_residual.data(), 1, &zero, d_adj_scaled_solution.data(), 1
    );
    add_alpha_solution_adjoint_kernel<<<
        (alpha.column_count + threads - 1) / threads, threads
    >>>(
        d_adj_solution.data(), alpha.scales.data(), alpha.column_count,
        d_adj_scaled_solution.data()
    );
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasScopy(
            blas, alpha.column_count, d_adj_scaled_solution.data(), 1,
            d_multiplier.data(), 1
        );
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasStrsv(
            blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
            alpha.column_count, alpha.qr_matrix.data(), alpha.qr_rows,
            d_multiplier.data(), 1
        );
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasStrsv(
            blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
            alpha.column_count, alpha.qr_matrix.data(), alpha.qr_rows,
            d_multiplier.data(), 1
        );
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasSgemv(
            blas, CUBLAS_OP_N, alpha.fit_count, alpha.column_count,
            &one, alpha.matrix_reference.data(), alpha.fit_count,
            d_multiplier.data(), 1, &zero, d_matrix_multiplier.data(), 1
        );
    }
    if (status == CUBLAS_STATUS_SUCCESS) {
        status = cublasSaxpy(
            blas, alpha.fit_count, &one, d_matrix_multiplier.data(), 1,
            d_adj_rhs.data(), 1
        );
    }
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(blas);
        fail_result(&result, "G3 alpha QR triangular adjoint failed");
        return false;
    }
    alpha_column_dot_kernel<<<alpha.column_count, threads, threads * sizeof(double)>>>(
        alpha.matrix_reference.data(), alpha.fit_count, d_adj_residual.data(),
        alpha.residual.data(), d_multiplier.data(), d_matrix_multiplier.data(),
        alpha.scaled_solution.data(), d_dot_adjoint_matrix.data()
    );
    alpha_row_adjoint_kernel<<<(alpha.fit_count + threads - 1) / threads, threads>>>(
        alpha.matrix_reference.data(), alpha.fit_count, alpha.column_count,
        d_adj_residual.data(), alpha.residual.data(), d_adj_rhs.data(),
        d_multiplier.data(), d_matrix_multiplier.data(), alpha.scaled_solution.data(),
        d_adj_solution.data(), alpha.scales.data(), d_dot_adjoint_matrix.data(),
        d_mode_m.data(), d_mode_n.data(), d_mode_kind.data(), d_radial_coefficients.data(),
        alpha.mode_count, config.alpha_radial_order, config.iota_degree, nfp,
        alpha.rho.data(), alpha.theta.data(), alpha.phi.data(), alpha.b_theta.data(),
        alpha.b_phi.data(), alpha.weights.data(), d_adj_weight.data(),
        d_adj_b_theta.data(), d_adj_b_phi.data()
    );
    float adj_weight_dot_weight = 0.0f;
    status = cublasSdot(
        blas, alpha.fit_count, d_adj_weight.data(), 1,
        alpha.weights.data(), 1, &adj_weight_dot_weight
    );
    cublasDestroy(blas);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fail_result(&result, "G3 alpha weight normalization adjoint failed");
        return false;
    }
    DeviceBuffer<float> d_adj_B(static_cast<size_t>(cache.points.count) * 3);
    DeviceBuffer<float> d_adj_grad_B(static_cast<size_t>(cache.points.count) * 9);
    DeviceBuffer<double> d_adj_G(1);
    if (!d_adj_B.data() || !d_adj_grad_B.data() || !d_adj_G.data() ||
        cudaMemset(d_adj_B.data(), 0, d_adj_B.size() * sizeof(float)) != cudaSuccess ||
        cudaMemset(d_adj_grad_B.data(), 0, d_adj_grad_B.size() * sizeof(float)) != cudaSuccess ||
        cudaMemset(d_adj_G.data(), 0, sizeof(double)) != cudaSuccess) {
        fail_result(&result, "G3 alpha field-adjoint allocation failed");
        return false;
    }
    const float weight_scale = static_cast<float>(
        std::sqrt(alpha.fit_count / std::max(alpha.weight_norm2, 1.0e-300))
    );
    alpha_B_adjoint_kernel<<<(alpha.fit_count + threads - 1) / threads, threads>>>(
        cache.B.data(), cache.points.grad_s.data(), cache.points.grad_theta.data(),
        cache.points.grad_phi.data(), cache.points.count, alpha.fit_count,
        alpha.weights.data(), d_adj_weight.data(), d_adj_b_theta.data(), d_adj_b_phi.data(),
        weight_scale, adj_weight_dot_weight, adj_normal,
        static_cast<float>(alpha.normal_B_relative_l2),
        static_cast<float>(alpha.normal_numerator),
        static_cast<float>(alpha.field_denominator), d_adj_B.data()
    );
    if (!cuda_stage_ok(cudaDeviceSynchronize(), result, "G3 alpha/iota point VJP")) return false;
    output.point_vjp_s = seconds_since(started);
    return map_fixed_front_field_adjoint(
        cache, d_adj_B, d_adj_grad_B, d_adj_G,
        coeffs_x, coeffs_y, coeffs_z, currents_a,
        n_base_coils, n_coeff, nfp, config, result, output
    );
}

}  // namespace

extern "C" {

std::size_t sgpu_score_config_size() {
    return sizeof(SgpuScoreConfig);
}

std::size_t sgpu_score_result_size() {
    return sizeof(SgpuScoreResult);
}

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
    config->axis_fallback_grid = 64;
    config->axis_max_candidates = 16;
    config->axis_fallback_max_candidates = 48;
    config->axis_newton_iters = 6;
    config->axis_fallback_newton_iters = 6;
    config->axis_trace_steps = 960;
    config->axis_sample_count = 240;
    config->axis_fallback_max_nfp = 7;
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
    config->psi_solver_mode = 2;
    config->psi_precision_mode = 2;
    config->psi_a = 0.05;
    config->psi_rho_min = 0.002;
    config->psi_ridge = 1.0e-6;
    const double default_levels[] = {
        0.001, 0.002, 0.004, 0.008, 0.02, 0.04, 0.08,
        0.16, 0.25, 0.36, 0.49, 0.64, 0.81,
    };
    config->surface_level_count = static_cast<int>(sizeof(default_levels) / sizeof(default_levels[0]));
    std::copy(default_levels, default_levels + config->surface_level_count, config->surface_levels);
    config->surface_theta_count = 256;
    config->surface_trace_steps = 800;
    config->surface_newton_iters = 20;
    config->surface_newton_tolerance = 1.0e-12;
    config->surface_max_radius_scale = 1.0;
    config->surface_drift_relative_tolerance = 0.05;
    config->surface_drift_absolute_tolerance = 5.0e-4;
    config->surface_long_trace_periods = 16;
    config->surface_long_trace_relative_tolerance = 0.05;
    config->flux_level_count = 11;
    config->flux_phi_count = 8;
    config->flux_theta_count = 256;
    config->flux_radial_quadrature = 24;
    config->flux_polynomial_degree = 4;
    config->flux_boundary_tolerance = 2.0e-6;
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
    config->alpha_solver_mode = 2;
    config->volume_rho_min = 0.08;
    config->alpha_ridge = 1.0e-7;
    const double weights[] = {10.0, 10.0, 10.0, 10.0, 42.0, 10.0, 8.0};
    std::copy(weights, weights + SGPU_SCORE_COMPONENT_COUNT, config->score_weights);
    config->score_axis_residual_scale = 1.0e-5;
    config->score_psi_angle_p95_scale = 3.0e-3;
    config->score_psi_angle_l2_scale = 1.0e-3;
    config->score_surface_inverse_aspect_saturation = 0.03;
    config->score_surface_drift_scale = 0.02;
    config->score_flux_section_std_scale = 0.01;
    config->score_flux_boundary_residual_scale = 2.0e-6;
    config->score_alpha_normal_B_scale = 1.0e-4;
    config->score_alpha_relative_l2_scale = 0.25;
    config->score_qs_global_scale = 0.05;
    config->score_qs_edge_scale = 0.07;
    config->score_qh_iota_threshold = 1.0;
    config->score_qh_iota_power = 2.0;
    config->score_volume_qs_size_floor = 0.65;
    config->score_volume_qs_iota_floor = 0.50;
    config->score_qh_total_iota_floor = 0.10;
    config->score_qh_total_helicity_floor = 0.10;
    config->score_qh_helicity_bad = 0.10;
    config->score_qh_helicity_good = 0.30;
    config->score_qh_helicity_exploration_fraction = 0.20;
    config->surface_selection_mode = 0;
    config->surface_confidence_periods = 2;
    config->surface_flux_bisection_iters = 6;
    config->surface_confidence_drift_center = 0.05;
    config->surface_confidence_drift_temperature = 0.0125;
    config->surface_confidence_smoothmax_temperature = 0.005;
    config->surface_confidence_minimum = 0.05;
    config->axis_hint_enabled = 0;
    config->axis_hint_require_continuation = 0;
    config->axis_hint_R = 0.0;
    config->axis_hint_Z = 0.0;
    config->axis_hint_max_distance = 0.10;
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
        double axis_trace_time = 0.0;
        if (!find_axis_native(
                field, coeffs_x, coeffs_y, coeffs_z, n_base_coils, n_coeff,
                nfp, *config, axis, axis_trace_time, result->timings)) {
            return_code = fail_from_backend(result, "axis search");
            break;
        }
        result->timings[SGPU_SCORE_TIME_AXIS_TRACE] = axis_trace_time;
        result->timings[SGPU_SCORE_TIME_AXIS_SEARCH] =
            std::max(0.0, seconds_since(stage_started) - axis_trace_time);
        result->axis_candidate_count = axis.candidate_count;
        result->axis_used_hint = axis.used_hint ? 1 : 0;
        result->axis_hint_distance = axis.hint_distance;
        if (axis.R.empty()) {
            result->status = axis.branch_lost ? SGPU_SCORE_BRANCH_LOST : SGPU_SCORE_NO_AXIS;
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
        const bool screen_ok = config->surface_selection_mode == 1
            ? screen_surface_confidence_native(
                field, axis, psi, nfp, *config, screens, *result)
            : screen_surfaces_native(
                field, axis, psi, nfp, *config, screens, result->timings);
        if (!screen_ok) {
            return_code = fail_from_backend(result, "surface screen");
            break;
        }
        result->timings[SGPU_SCORE_TIME_SURFACE_SCREEN] = seconds_since(stage_started);
        const auto best_strict = std::max_element(screens.begin(), screens.end(), [](const auto& lhs, const auto& rhs) {
            return (lhs.strict && lhs.verified ? lhs.level : -1.0) <
                   (rhs.strict && rhs.verified ? rhs.level : -1.0);
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
        if (best_strict == screens.end() || !best_strict->strict || !best_strict->verified) {
            result->status = SGPU_SCORE_DRIFT_REJECTED;
            fill_early_components(*config, coil, axis, psi, screens, *result);
            break;
        }
        const bool downstream_ok = run_downstream_gpu(
                field, currents_a, n_base_coils, nfp, axis, psi,
                screens, *config, *result, config->surface_selection_mode == 0);
        fill_early_components(*config, coil, axis, psi, screens, *result);
        if (!downstream_ok) {
            if (result->status == SGPU_SCORE_INTERNAL_ERROR) return_code = 1;
            break;
        }
        if (g_active_g2_cache && g_active_g2_cache->ready) {
            g_active_g2_cache->axis = axis;
        }
    } while (false);

    if (g_active_g2_cache && g_active_g2_cache->ready) {
        g_active_g2_cache->field = field;
        field = nullptr;
    }
    if (field) sgpu_destroy_field(field);
    stage_started = Clock::now();
    finalize_score(*config, *result);
    result->timings[SGPU_SCORE_TIME_SCORE] = seconds_since(stage_started);
    result->timings[SGPU_SCORE_TIME_TOTAL] = seconds_since(total_started);
    return return_code;
}

std::size_t sgpu_score_gradient_result_size() {
    return sizeof(SgpuScoreGradientResult);
}

int sgpu_coil_component_gradient(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    double* component_value,
    double* gradient_x,
    double* gradient_y,
    double* gradient_z,
    double* gradient_current
) {
    if (!component_value || !gradient_x || !gradient_y || !gradient_z || !gradient_current) {
        sgpu_internal_set_error("coil-component gradient output pointer is null");
        return 1;
    }
    CoilComponentGradient gradient;
    std::string error;
    if (!compute_coil_component_gradient_impl(
            coeffs_x, coeffs_y, coeffs_z, currents_a,
            n_base_coils, n_coeff, nfp, gradient, error)) {
        sgpu_internal_set_error(error.c_str());
        return 1;
    }
    *component_value = gradient.value;
    std::copy(gradient.x.begin(), gradient.x.end(), gradient_x);
    std::copy(gradient.y.begin(), gradient.y.end(), gradient_y);
    std::copy(gradient.z.begin(), gradient.z.end(), gradient_z);
    std::copy(gradient.current.begin(), gradient.current.end(), gradient_current);
    sgpu_internal_set_error("");
    return 0;
}

int sgpu_score_coils_g1_gradient(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig* config,
    SgpuScoreResult* score_result,
    double* gradient_x,
    double* gradient_y,
    double* gradient_z,
    double* gradient_current,
    SgpuScoreGradientResult* gradient_result
) {
    if (!gradient_result) {
        sgpu_internal_set_error("score gradient result pointer is null");
        return 1;
    }
    std::memset(gradient_result, 0, sizeof(*gradient_result));
    gradient_result->abi_version = SGPU_SCORE_GRADIENT_ABI_VERSION;
    gradient_result->struct_size = sizeof(*gradient_result);
    gradient_result->status = 1;
    gradient_result->gradient_group = 1;
    if (!score_result || !gradient_x || !gradient_y || !gradient_z || !gradient_current || !config) {
        std::snprintf(
            gradient_result->error_message,
            sizeof(gradient_result->error_message),
            "%s",
            "G1 gradient input/output pointer is null"
        );
        sgpu_internal_set_error(gradient_result->error_message);
        return 1;
    }
    const auto forward_started = Clock::now();
    const int score_code = sgpu_score_coils(
        coeffs_x, coeffs_y, coeffs_z, currents_a,
        n_base_coils, n_coeff, nfp, config, score_result
    );
    gradient_result->forward_wall_s = seconds_since(forward_started);
    if (score_code != 0) {
        std::snprintf(
            gradient_result->error_message,
            sizeof(gradient_result->error_message),
            "forward score failed: %.220s",
            score_result->error_message
        );
        sgpu_internal_set_error(gradient_result->error_message);
        return score_code;
    }
    const auto gradient_started = Clock::now();
    CoilComponentGradient component_gradient;
    std::string error;
    if (!compute_coil_component_gradient_impl(
            coeffs_x, coeffs_y, coeffs_z, currents_a,
            n_base_coils, n_coeff, nfp, component_gradient, error)) {
        std::snprintf(
            gradient_result->error_message,
            sizeof(gradient_result->error_message),
            "%s",
            error.c_str()
        );
        sgpu_internal_set_error(gradient_result->error_message);
        return 1;
    }
    double total_weight = 0.0;
    for (int component = 0; component < SGPU_SCORE_COMPONENT_COUNT; ++component) {
        total_weight += std::max(config->score_weights[component], 0.0);
    }
    const double gate = score_result->score_qh_total_iota_factor *
        score_result->score_qh_total_helicity_factor;
    const double score_scale = total_weight > 0.0
        ? std::max(config->score_weights[SGPU_SCORE_COMPONENT_COIL], 0.0) / total_weight * gate
        : 0.0;
    double score_norm2 = 0.0;
    double component_norm2 = 0.0;
    size_t gradient_count = 0;
    const int parameter_count = n_base_coils * n_coeff;
    const std::vector<double>* sources[] = {
        &component_gradient.x, &component_gradient.y, &component_gradient.z,
    };
    double* destinations[] = {gradient_x, gradient_y, gradient_z};
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        for (int parameter = 0; parameter < parameter_count; ++parameter) {
            const double component_value = (*sources[coordinate])[parameter];
            destinations[coordinate][parameter] = score_scale * component_value;
            component_norm2 += component_value * component_value;
            score_norm2 += destinations[coordinate][parameter] * destinations[coordinate][parameter];
            ++gradient_count;
        }
    }
    for (int coil = 0; coil < n_base_coils; ++coil) {
        gradient_current[coil] = score_scale * component_gradient.current[coil];
        component_norm2 += component_gradient.current[coil] * component_gradient.current[coil];
        score_norm2 += gradient_current[coil] * gradient_current[coil];
        ++gradient_count;
    }
    gradient_result->gradient_wall_s = seconds_since(gradient_started);
    gradient_result->score_gradient_rms = std::sqrt(score_norm2 / std::max<size_t>(gradient_count, 1));
    gradient_result->coil_component_gradient_rms =
        std::sqrt(component_norm2 / std::max<size_t>(gradient_count, 1));
    gradient_result->status = 0;
    sgpu_internal_set_error("");
    return 0;
}

int sgpu_score_coils_g2_gradient(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig* config,
    SgpuScoreResult* score_result,
    double* gradient_x,
    double* gradient_y,
    double* gradient_z,
    double* gradient_current,
    SgpuScoreGradientResult* gradient_result
) {
    if (!gradient_result) {
        sgpu_internal_set_error("score gradient result pointer is null");
        return 1;
    }
    std::memset(gradient_result, 0, sizeof(*gradient_result));
    gradient_result->abi_version = SGPU_SCORE_GRADIENT_ABI_VERSION;
    gradient_result->struct_size = sizeof(*gradient_result);
    gradient_result->status = 1;
    gradient_result->gradient_group = 2;
    if (!score_result || !gradient_x || !gradient_y || !gradient_z ||
        !gradient_current || !config || !coeffs_x || !coeffs_y || !coeffs_z ||
        !currents_a) {
        std::snprintf(
            gradient_result->error_message,
            sizeof(gradient_result->error_message),
            "%s",
            "G2 gradient input/output pointer is null"
        );
        sgpu_internal_set_error(gradient_result->error_message);
        return 1;
    }
    FixedFrontG2Cache cache;
    const auto forward_started = Clock::now();
    g_active_gradient_group = 2;
    g_active_g2_cache = &cache;
    const int score_code = sgpu_score_coils(
        coeffs_x, coeffs_y, coeffs_z, currents_a,
        n_base_coils, n_coeff, nfp, config, score_result
    );
    g_active_g2_cache = nullptr;
    g_active_gradient_group = 0;
    gradient_result->forward_wall_s = seconds_since(forward_started);
    if (score_code != 0) {
        if (cache.field) sgpu_destroy_field(cache.field);
        std::snprintf(
            gradient_result->error_message,
            sizeof(gradient_result->error_message),
            "G2 score forward failed: %.210s",
            score_result->error_message
        );
        sgpu_internal_set_error(gradient_result->error_message);
        return score_code;
    }
    if (score_result->status != SGPU_SCORE_OK) {
        if (cache.field) sgpu_destroy_field(cache.field);
        const size_t parameter_count =
            static_cast<size_t>(n_base_coils) * static_cast<size_t>(n_coeff);
        std::fill_n(gradient_x, parameter_count, 0.0);
        std::fill_n(gradient_y, parameter_count, 0.0);
        std::fill_n(gradient_z, parameter_count, 0.0);
        std::fill_n(gradient_current, static_cast<size_t>(n_base_coils), 0.0);
        gradient_result->status = 1;
        sgpu_internal_set_error("");
        return 0;
    }
    if (!cache.ready || !cache.field) {
        if (cache.field) sgpu_destroy_field(cache.field);
        std::snprintf(
            gradient_result->error_message,
            sizeof(gradient_result->error_message),
            "%s",
            "G2 cache is incomplete after an ok score"
        );
        sgpu_internal_set_error(gradient_result->error_message);
        return 1;
    }
    const auto gradient_started = Clock::now();
    CoilComponentGradient g1;
    FixedFrontG2Gradient g2;
    std::string error;
    if (!compute_coil_component_gradient_impl(
            coeffs_x, coeffs_y, coeffs_z, currents_a,
            n_base_coils, n_coeff, nfp, g1, error) ||
        !compute_fixed_front_g2_gradient(
            cache, coeffs_x, coeffs_y, coeffs_z, currents_a,
            n_base_coils, n_coeff, nfp, *config, *score_result, g2)) {
        sgpu_destroy_field(cache.field);
        if (!score_result->error_message[0]) {
            std::snprintf(
                gradient_result->error_message,
                sizeof(gradient_result->error_message),
                "%s",
                error.c_str()
            );
        } else {
            std::snprintf(
                gradient_result->error_message,
                sizeof(gradient_result->error_message),
                "%.250s",
                score_result->error_message
            );
        }
        sgpu_internal_set_error(gradient_result->error_message);
        return 1;
    }
    sgpu_destroy_field(cache.field);
    cache.field = nullptr;

    double total_weight = 0.0;
    for (int component = 0; component < SGPU_SCORE_COMPONENT_COUNT; ++component) {
        total_weight += std::max(config->score_weights[component], 0.0);
    }
    const double g1_scale = total_weight > 0.0
        ? std::max(config->score_weights[SGPU_SCORE_COMPONENT_COIL], 0.0) / total_weight *
            score_result->score_qh_total_iota_factor * score_result->score_qh_total_helicity_factor
        : 0.0;
    const int parameter_count = n_base_coils * n_coeff;
    const std::vector<double>* g1_sources[] = {&g1.x, &g1.y, &g1.z};
    const std::vector<double>* g2_sources[] = {&g2.x, &g2.y, &g2.z};
    double* destinations[] = {gradient_x, gradient_y, gradient_z};
    double score_norm2 = 0.0;
    double component_norm2 = 0.0;
    size_t gradient_count = 0;
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        for (int parameter = 0; parameter < parameter_count; ++parameter) {
            const double g1_value = (*g1_sources[coordinate])[parameter];
            destinations[coordinate][parameter] =
                g1_scale * g1_value + (*g2_sources[coordinate])[parameter];
            score_norm2 += destinations[coordinate][parameter] * destinations[coordinate][parameter];
            component_norm2 += g1_value * g1_value;
            ++gradient_count;
        }
    }
    for (int coil = 0; coil < n_base_coils; ++coil) {
        gradient_current[coil] = g1_scale * g1.current[coil] + g2.current[coil];
        score_norm2 += gradient_current[coil] * gradient_current[coil];
        component_norm2 += g1.current[coil] * g1.current[coil];
        ++gradient_count;
    }
    gradient_result->point_vjp_s = g2.point_vjp_s;
    gradient_result->field_vjp_s = g2.field_vjp_s;
    gradient_result->parameter_map_s = g2.parameter_map_s;
    gradient_result->gradient_wall_s = seconds_since(gradient_started);
    gradient_result->score_gradient_rms =
        std::sqrt(score_norm2 / std::max<size_t>(gradient_count, 1));
    gradient_result->coil_component_gradient_rms =
        std::sqrt(component_norm2 / std::max<size_t>(gradient_count, 1));
    gradient_result->status = 0;
    sgpu_internal_set_error("");
    return 0;
}

int sgpu_score_coils_g2_frozen_batch(
    const double* center_coeffs_x,
    const double* center_coeffs_y,
    const double* center_coeffs_z,
    const double* center_currents_a,
    const double* query_coeffs_x,
    const double* query_coeffs_y,
    const double* query_coeffs_z,
    const double* query_currents_a,
    int query_count,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig* config,
    SgpuScoreResult* center_score_result,
    double* frozen_scores,
    double* volume_components,
    double* coil_components,
    double* target_errors,
    double* qa_errors,
    double* qp_errors
) {
    if (!center_coeffs_x || !center_coeffs_y || !center_coeffs_z ||
        !center_currents_a || !query_coeffs_x || !query_coeffs_y ||
        !query_coeffs_z || !query_currents_a || query_count <= 0 ||
        n_base_coils <= 0 || n_coeff <= 0 || nfp <= 0 || !config ||
        !center_score_result || !frozen_scores || !volume_components ||
        !coil_components || !target_errors || !qa_errors || !qp_errors) {
        sgpu_internal_set_error("invalid G2 frozen-batch input");
        return 1;
    }
    FixedFrontG2Cache cache;
    g_active_gradient_group = 2;
    g_active_g2_cache = &cache;
    const int score_code = sgpu_score_coils(
        center_coeffs_x, center_coeffs_y, center_coeffs_z, center_currents_a,
        n_base_coils, n_coeff, nfp, config, center_score_result
    );
    g_active_g2_cache = nullptr;
    g_active_gradient_group = 0;
    if (score_code != 0 || center_score_result->status != SGPU_SCORE_OK ||
        !cache.ready || !cache.field) {
        if (cache.field) sgpu_destroy_field(cache.field);
        if (score_code == 0 && center_score_result->status != SGPU_SCORE_OK) {
            sgpu_internal_set_error("G2 frozen-batch center score is not ok");
            return 1;
        }
        if (score_code == 0) {
            sgpu_internal_set_error("G2 frozen-batch center cache is incomplete");
            return 1;
        }
        return score_code;
    }

    const size_t coefficient_stride =
        static_cast<size_t>(n_base_coils) * static_cast<size_t>(n_coeff);
    const size_t current_stride = static_cast<size_t>(n_base_coils);
    std::string error;
    for (int query = 0; query < query_count; ++query) {
        if (!evaluate_fixed_front_g2_scalar(
                cache,
                query_coeffs_x + query * coefficient_stride,
                query_coeffs_y + query * coefficient_stride,
                query_coeffs_z + query * coefficient_stride,
                query_currents_a + query * current_stride,
                n_base_coils, n_coeff, nfp, *config, *center_score_result,
                frozen_scores[query], volume_components[query],
                coil_components[query], target_errors[query],
                qa_errors[query], qp_errors[query], error)) {
            sgpu_destroy_field(cache.field);
            std::snprintf(
                center_score_result->error_message,
                sizeof(center_score_result->error_message),
                "G2 frozen query %d failed: %.215s",
                query,
                error.c_str()
            );
            center_score_result->status = SGPU_SCORE_INTERNAL_ERROR;
            sgpu_internal_set_error(center_score_result->error_message);
            return 1;
        }
    }
    sgpu_destroy_field(cache.field);
    cache.field = nullptr;
    sgpu_internal_set_error("");
    return 0;
}

int sgpu_score_coils_g3_frozen_batch(
    const double* center_coeffs_x,
    const double* center_coeffs_y,
    const double* center_coeffs_z,
    const double* center_currents_a,
    const double* query_coeffs_x,
    const double* query_coeffs_y,
    const double* query_coeffs_z,
    const double* query_currents_a,
    int query_count,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig* config,
    SgpuScoreResult* center_score_result,
    double* frozen_scores,
    double* volume_components,
    double* coordinate_components,
    double* iota_components,
    double* coil_components,
    double* target_errors,
    double* qa_errors,
    double* qp_errors,
    double* iota_minima,
    double* iota_maxima
) {
    if (!center_coeffs_x || !center_coeffs_y || !center_coeffs_z ||
        !center_currents_a || !query_coeffs_x || !query_coeffs_y ||
        !query_coeffs_z || !query_currents_a || query_count <= 0 ||
        n_base_coils <= 0 || n_coeff <= 0 || nfp <= 0 || !config ||
        !center_score_result || !frozen_scores || !volume_components ||
        !coordinate_components || !iota_components || !coil_components ||
        !target_errors || !qa_errors || !qp_errors || !iota_minima ||
        !iota_maxima) {
        sgpu_internal_set_error("invalid G3 frozen-batch input");
        return 1;
    }
    FixedFrontG2Cache cache;
    g_active_gradient_group = 3;
    g_active_g2_cache = &cache;
    const int score_code = sgpu_score_coils(
        center_coeffs_x, center_coeffs_y, center_coeffs_z, center_currents_a,
        n_base_coils, n_coeff, nfp, config, center_score_result
    );
    g_active_g2_cache = nullptr;
    g_active_gradient_group = 0;
    if (score_code != 0 || center_score_result->status != SGPU_SCORE_OK ||
        !cache.ready || !cache.field || !cache.alpha.adjoint_ready) {
        if (cache.field) sgpu_destroy_field(cache.field);
        if (score_code == 0 && center_score_result->status != SGPU_SCORE_OK) {
            sgpu_internal_set_error("G3 frozen-batch center score is not ok");
            return 1;
        }
        if (score_code == 0) {
            sgpu_internal_set_error("G3 frozen-batch center cache is incomplete");
            return 1;
        }
        return score_code;
    }

    const size_t coefficient_stride =
        static_cast<size_t>(n_base_coils) * static_cast<size_t>(n_coeff);
    const size_t current_stride = static_cast<size_t>(n_base_coils);
    std::string error;
    for (int query = 0; query < query_count; ++query) {
        if (!evaluate_fixed_front_g3_scalar(
                cache,
                query_coeffs_x + query * coefficient_stride,
                query_coeffs_y + query * coefficient_stride,
                query_coeffs_z + query * coefficient_stride,
                query_currents_a + query * current_stride,
                n_base_coils, n_coeff, nfp, *config, *center_score_result,
                frozen_scores[query], volume_components[query],
                coordinate_components[query], iota_components[query],
                coil_components[query], target_errors[query], qa_errors[query],
                qp_errors[query], iota_minima[query], iota_maxima[query], error)) {
            sgpu_destroy_field(cache.field);
            std::snprintf(
                center_score_result->error_message,
                sizeof(center_score_result->error_message),
                "G3 frozen query %d failed: %.215s",
                query,
                error.c_str()
            );
            center_score_result->status = SGPU_SCORE_INTERNAL_ERROR;
            sgpu_internal_set_error(center_score_result->error_message);
            return 1;
        }
    }
    sgpu_destroy_field(cache.field);
    cache.field = nullptr;
    sgpu_internal_set_error("");
    return 0;
}

int sgpu_score_coils_g4_fixed_branch_batch(
    const double* center_coeffs_x,
    const double* center_coeffs_y,
    const double* center_coeffs_z,
    const double* center_currents_a,
    const double* query_coeffs_x,
    const double* query_coeffs_y,
    const double* query_coeffs_z,
    const double* query_currents_a,
    int query_count,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig* config,
    SgpuScoreResult* center_score_result,
    SgpuScoreResult* query_score_results
) {
    if (!center_coeffs_x || !center_coeffs_y || !center_coeffs_z ||
        !center_currents_a || !query_coeffs_x || !query_coeffs_y ||
        !query_coeffs_z || !query_currents_a || query_count <= 0 ||
        n_base_coils <= 0 || n_coeff <= 0 || nfp <= 0 || !config ||
        !center_score_result || !query_score_results) {
        sgpu_internal_set_error("invalid G4 fixed-branch batch input");
        return 1;
    }
    FixedFrontG2Cache cache;
    g_active_gradient_group = 2;
    g_active_g2_cache = &cache;
    const int score_code = sgpu_score_coils(
        center_coeffs_x, center_coeffs_y, center_coeffs_z, center_currents_a,
        n_base_coils, n_coeff, nfp, config, center_score_result
    );
    g_active_g2_cache = nullptr;
    g_active_gradient_group = 0;
    if (score_code != 0 || center_score_result->status != SGPU_SCORE_OK ||
        !cache.ready || cache.axis.R.empty()) {
        if (cache.field) sgpu_destroy_field(cache.field);
        sgpu_internal_set_error("G4 fixed-branch center score is not ok");
        return score_code != 0 ? score_code : 1;
    }
    if (cache.field) {
        sgpu_destroy_field(cache.field);
        cache.field = nullptr;
    }

    const size_t coefficient_stride = static_cast<size_t>(n_base_coils) * n_coeff;
    for (int query_index = 0; query_index < query_count; ++query_index) {
        std::string error;
        if (!evaluate_fixed_branch_g4_scalar(
                cache,
                query_coeffs_x + query_index * coefficient_stride,
                query_coeffs_y + query_index * coefficient_stride,
                query_coeffs_z + query_index * coefficient_stride,
                query_currents_a + static_cast<size_t>(query_index) * n_base_coils,
                n_base_coils, n_coeff, nfp, *config, *center_score_result,
                query_score_results[query_index], error)) {
            std::snprintf(
                center_score_result->error_message,
                sizeof(center_score_result->error_message),
                "G4 fixed-branch query %d failed: %.205s",
                query_index,
                error.c_str()
            );
            center_score_result->status = SGPU_SCORE_INTERNAL_ERROR;
            sgpu_internal_set_error(center_score_result->error_message);
            return 1;
        }
    }
    sgpu_internal_set_error("");
    return 0;
}

int sgpu_score_coils_g3_gradient(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    const SgpuScoreConfig* config,
    SgpuScoreResult* score_result,
    double* gradient_x,
    double* gradient_y,
    double* gradient_z,
    double* gradient_current,
    SgpuScoreGradientResult* gradient_result
) {
    if (!gradient_result) {
        sgpu_internal_set_error("score gradient result pointer is null");
        return 1;
    }
    std::memset(gradient_result, 0, sizeof(*gradient_result));
    gradient_result->abi_version = SGPU_SCORE_GRADIENT_ABI_VERSION;
    gradient_result->struct_size = sizeof(*gradient_result);
    gradient_result->status = 1;
    gradient_result->gradient_group = 3;
    if (!score_result || !gradient_x || !gradient_y || !gradient_z ||
        !gradient_current || !config || !coeffs_x || !coeffs_y || !coeffs_z ||
        !currents_a) {
        std::snprintf(
            gradient_result->error_message,
            sizeof(gradient_result->error_message),
            "%s",
            "G3 gradient input/output pointer is null"
        );
        sgpu_internal_set_error(gradient_result->error_message);
        return 1;
    }
    FixedFrontG2Cache cache;
    const auto forward_started = Clock::now();
    g_active_gradient_group = 3;
    g_active_g2_cache = &cache;
    const int score_code = sgpu_score_coils(
        coeffs_x, coeffs_y, coeffs_z, currents_a,
        n_base_coils, n_coeff, nfp, config, score_result
    );
    g_active_g2_cache = nullptr;
    g_active_gradient_group = 0;
    gradient_result->forward_wall_s = seconds_since(forward_started);
    if (score_code != 0 || score_result->status != SGPU_SCORE_OK || !cache.ready ||
        !cache.field || !cache.alpha.adjoint_ready) {
        if (cache.field) sgpu_destroy_field(cache.field);
        std::snprintf(
            gradient_result->error_message,
            sizeof(gradient_result->error_message),
            "G3 requires an ok complete QR score: %.180s",
            score_result->error_message
        );
        sgpu_internal_set_error(gradient_result->error_message);
        return score_code != 0 ? score_code : 1;
    }
    const auto gradient_started = Clock::now();
    CoilComponentGradient g1;
    FixedFrontG2Gradient g2;
    FixedFrontG2Gradient g3;
    std::string error;
    if (!compute_coil_component_gradient_impl(
            coeffs_x, coeffs_y, coeffs_z, currents_a,
            n_base_coils, n_coeff, nfp, g1, error) ||
        !compute_fixed_front_g2_gradient(
            cache, coeffs_x, coeffs_y, coeffs_z, currents_a,
            n_base_coils, n_coeff, nfp, *config, *score_result, g2) ||
        !compute_alpha_iota_g3_gradient(
            cache, coeffs_x, coeffs_y, coeffs_z, currents_a,
            n_base_coils, n_coeff, nfp, *config, *score_result, g3)) {
        sgpu_destroy_field(cache.field);
        if (!score_result->error_message[0]) {
            std::snprintf(
                gradient_result->error_message,
                sizeof(gradient_result->error_message),
                "%s",
                error.c_str()
            );
        } else {
            std::snprintf(
                gradient_result->error_message,
                sizeof(gradient_result->error_message),
                "%.250s",
                score_result->error_message
            );
        }
        sgpu_internal_set_error(gradient_result->error_message);
        return 1;
    }
    sgpu_destroy_field(cache.field);
    cache.field = nullptr;

    double total_weight = 0.0;
    for (int component = 0; component < SGPU_SCORE_COMPONENT_COUNT; ++component) {
        total_weight += std::max(config->score_weights[component], 0.0);
    }
    const double g1_scale = total_weight > 0.0
        ? std::max(config->score_weights[SGPU_SCORE_COMPONENT_COIL], 0.0) / total_weight *
            score_result->score_qh_total_iota_factor * score_result->score_qh_total_helicity_factor
        : 0.0;
    const int parameter_count = n_base_coils * n_coeff;
    const std::vector<double>* g1_sources[] = {&g1.x, &g1.y, &g1.z};
    const std::vector<double>* g2_sources[] = {&g2.x, &g2.y, &g2.z};
    const std::vector<double>* g3_sources[] = {&g3.x, &g3.y, &g3.z};
    double* destinations[] = {gradient_x, gradient_y, gradient_z};
    double score_norm2 = 0.0;
    double component_norm2 = 0.0;
    size_t gradient_count = 0;
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
        for (int parameter = 0; parameter < parameter_count; ++parameter) {
            const double g1_value = (*g1_sources[coordinate])[parameter];
            destinations[coordinate][parameter] =
                g1_scale * g1_value + (*g2_sources[coordinate])[parameter] +
                (*g3_sources[coordinate])[parameter];
            score_norm2 += destinations[coordinate][parameter] * destinations[coordinate][parameter];
            component_norm2 += g1_value * g1_value;
            ++gradient_count;
        }
    }
    for (int coil = 0; coil < n_base_coils; ++coil) {
        gradient_current[coil] = g1_scale * g1.current[coil] + g2.current[coil] + g3.current[coil];
        score_norm2 += gradient_current[coil] * gradient_current[coil];
        component_norm2 += g1.current[coil] * g1.current[coil];
        ++gradient_count;
    }
    gradient_result->point_vjp_s = g2.point_vjp_s + g3.point_vjp_s;
    gradient_result->field_vjp_s = g2.field_vjp_s + g3.field_vjp_s;
    gradient_result->parameter_map_s = g2.parameter_map_s + g3.parameter_map_s;
    gradient_result->gradient_wall_s = seconds_since(gradient_started);
    gradient_result->score_gradient_rms =
        std::sqrt(score_norm2 / std::max<size_t>(gradient_count, 1));
    gradient_result->coil_component_gradient_rms =
        std::sqrt(component_norm2 / std::max<size_t>(gradient_count, 1));
    gradient_result->status = 0;
    sgpu_internal_set_error("");
    return 0;
}

}  // extern "C"
