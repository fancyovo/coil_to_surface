#pragma once

#include <cstddef>
#include <cstdint>

#define SGPU_SCORE_ABI_VERSION 1u
#define SGPU_SCORE_MAX_SURFACE_LEVELS 16
#define SGPU_SCORE_COMPONENT_COUNT 6
#define SGPU_SCORE_TIMING_COUNT 16

enum SgpuScoreStatus {
    SGPU_SCORE_OK = 0,
    SGPU_SCORE_NO_AXIS = 1,
    SGPU_SCORE_NO_SURFACE = 2,
    SGPU_SCORE_DRIFT_REJECTED = 3,
    SGPU_SCORE_FLUX_REJECTED = 4,
    SGPU_SCORE_ALPHA_FAILED = 5,
    SGPU_SCORE_INTERNAL_ERROR = 100,
};

enum SgpuScoreComponent {
    SGPU_SCORE_COMPONENT_AXIS = 0,
    SGPU_SCORE_COMPONENT_PSI = 1,
    SGPU_SCORE_COMPONENT_SURFACE = 2,
    SGPU_SCORE_COMPONENT_COORDINATE = 3,
    SGPU_SCORE_COMPONENT_VOLUME_QS = 4,
    SGPU_SCORE_COMPONENT_COIL = 5,
};

enum SgpuScoreTiming {
    SGPU_SCORE_TIME_TOTAL = 0,
    SGPU_SCORE_TIME_FIELD_CREATE = 1,
    SGPU_SCORE_TIME_COIL_GEOMETRY = 2,
    SGPU_SCORE_TIME_AXIS_SEARCH = 3,
    SGPU_SCORE_TIME_AXIS_TRACE = 4,
    SGPU_SCORE_TIME_PSI_POINTS = 5,
    SGPU_SCORE_TIME_PSI_FIT = 6,
    SGPU_SCORE_TIME_PSI_VALIDATE = 7,
    SGPU_SCORE_TIME_SURFACE_SCREEN = 8,
    SGPU_SCORE_TIME_FLUX = 9,
    SGPU_SCORE_TIME_VOLUME_POINTS = 10,
    SGPU_SCORE_TIME_FIELD_VOLUME = 11,
    SGPU_SCORE_TIME_ALPHA_ASSEMBLE = 12,
    SGPU_SCORE_TIME_ALPHA_QR = 13,
    SGPU_SCORE_TIME_QS_METRICS = 14,
    SGPU_SCORE_TIME_SCORE = 15,
};

struct SgpuScoreConfig {
    std::uint32_t abi_version;
    std::uint32_t struct_size;

    std::int32_t device_id;
    std::int32_t segments_per_coil;
    std::int32_t target_M;
    std::int32_t target_N;

    std::int32_t axis_grid;
    std::int32_t axis_fallback_grid;
    std::int32_t axis_max_candidates;
    std::int32_t axis_fallback_max_candidates;
    std::int32_t axis_newton_iters;
    std::int32_t axis_fallback_newton_iters;
    std::int32_t axis_trace_steps;
    std::int32_t axis_sample_count;
    double axis_span;
    double axis_tolerance;
    double axis_r_floor;
    double axis_fd_relative;
    double axis_fd_absolute;
    double axis_topology_margin;

    std::int32_t psi_poly_degree;
    std::int32_t psi_m_tor;
    std::int32_t psi_n_r;
    std::int32_t psi_n_z;
    std::int32_t psi_n_phi;
    std::int32_t psi_validation_points;
    double psi_a;
    double psi_rho_min;
    double psi_ridge;

    std::int32_t surface_level_count;
    std::int32_t surface_theta_count;
    std::int32_t surface_trace_steps;
    std::int32_t surface_newton_iters;
    double surface_levels[SGPU_SCORE_MAX_SURFACE_LEVELS];
    double surface_newton_tolerance;
    double surface_max_radius_scale;
    double surface_drift_relative_tolerance;
    double surface_drift_absolute_tolerance;

    std::int32_t flux_level_count;
    std::int32_t flux_phi_count;
    std::int32_t flux_theta_count;
    std::int32_t flux_radial_quadrature;
    std::int32_t flux_polynomial_degree;
    double flux_boundary_tolerance;
    double flux_section_relative_std_tolerance;

    std::int32_t volume_point_count;
    std::int32_t volume_phi_count;
    std::int32_t volume_theta_count;
    std::int32_t alpha_fit_point_count;
    std::int32_t alpha_radial_order;
    std::int32_t alpha_poloidal_order;
    std::int32_t alpha_toroidal_order;
    std::int32_t iota_degree;
    std::int32_t radial_bin_count;
    double volume_rho_min;
    double alpha_ridge;

    double score_weights[SGPU_SCORE_COMPONENT_COUNT];
    double score_axis_residual_scale;
    double score_psi_angle_p95_scale;
    double score_psi_angle_l2_scale;
    double score_surface_inverse_aspect_scale;
    double score_surface_drift_scale;
    double score_flux_section_std_scale;
    double score_flux_boundary_residual_scale;
    double score_alpha_normal_B_scale;
    double score_alpha_relative_l2_scale;
    double score_qs_global_scale;
    double score_qs_edge_scale;
};

struct SgpuScoreResult {
    std::uint32_t abi_version;
    std::uint32_t struct_size;
    std::int32_t status;
    std::int32_t stage_completed;
    std::int32_t device_id;
    std::int32_t reserved_i32;

    double score;
    double components[SGPU_SCORE_COMPONENT_COUNT];
    double timings[SGPU_SCORE_TIMING_COUNT];

    double axis_R;
    double axis_Z;
    double axis_residual;
    double axis_topology_trace;
    double axis_topology_det;
    double axis_ellipse_aspect;

    double psi_train_rms;
    double psi_angle_mean;
    double psi_angle_p95;
    double psi_angle_l2;

    double surface_level;
    double surface_drift_relative_p95;
    double surface_effective_minor_radius;
    double surface_inverse_aspect_ratio;
    double surface_volume;

    double flux_edge;
    double flux_fit_relative_rms;
    double flux_section_relative_std_edge;
    double flux_boundary_residual_max;
    double flux_derivative_min;
    double flux_derivative_max;

    double alpha_relative_l2;
    double alpha_normal_B_relative_l2;
    double iota_min;
    double iota_max;

    double qs_global_error;
    double qs_edge_error;
    double qs_abs_p95;

    double coil_length_mean;
    double coil_curvature_p95;
    double coil_curvature_max;
    double coil_min_intercoil_distance;
    double coil_min_axis_distance;
    double coil_high_mode_energy_fraction;
    double coil_current_abs_max_a;

    std::int32_t axis_candidate_count;
    std::int32_t stable_surface_count;
    std::int32_t volume_point_count;
    std::int32_t alpha_column_count;
    char error_message[256];
};

extern "C" {

int sgpu_default_score_config(SgpuScoreConfig* config);

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
);

int sgpu_create_field(
    const double* coeffs_x,
    const double* coeffs_y,
    const double* coeffs_z,
    const double* currents_a,
    int n_base_coils,
    int n_coeff,
    int nfp,
    int segments_per_coil,
    int device_id,
    void** out_handle
);

void sgpu_destroy_field(void* handle);

int sgpu_segment_count(void* handle);

int sgpu_eval_B(
    void* handle,
    const double* xyz_host,
    double* B_host,
    int n_points
);

int sgpu_eval_B_f32(
    void* handle,
    const float* xyz_host,
    float* B_host,
    int n_points
);

int sgpu_eval_B_grad(
    void* handle,
    const double* xyz_host,
    double* B_host,
    double* grad_B_host,
    int n_points
);

int sgpu_eval_B_grad_f32(
    void* handle,
    const float* xyz_host,
    float* B_host,
    float* grad_B_host,
    int n_points
);

int sgpu_normal_eq(
    void* handle,
    const double* mat_host,
    const double* rhs_host,
    double* ata_host,
    double* atb_host,
    int n_rows,
    int n_cols
);

int sgpu_normal_eq_f32(
    void* handle,
    const float* mat_host,
    const float* rhs_host,
    float* ata_host,
    float* atb_host,
    int n_rows,
    int n_cols
);

int sgpu_fit_psi_fullgpu(
    void* handle,
    const double* R_host,
    const double* Z_host,
    const double* phi_host,
    int n_points,
    const double* axis_R_host,
    const double* axis_Z_host,
    const double* axis_R_phi_host,
    const double* axis_Z_phi_host,
    int n_axis,
    const int* mode_a_host,
    const int* mode_b_host,
    const int* mode_m_host,
    const int* mode_kind_host,
    int n_coeff,
    int nfp,
    double a,
    int poly_degree,
    int m_tor,
    double ridge,
    int solver_mode,
    int precision_mode,
    double* coeff_host,
    double* train_rms_out,
    double* stats_out,
    int stats_len
);

int sgpu_surface_points_from_level(
    const double* coeff_host,
    const int* mode_a_host,
    const int* mode_b_host,
    const int* mode_m_host,
    const int* mode_kind_host,
    int n_coeff,
    int nfp,
    double a,
    int poly_degree,
    int m_tor,
    const double* axis_R_host,
    const double* axis_Z_host,
    int n_axis,
    int order,
    double psi_level,
    int maxiter,
    double tol,
    double max_radius_scale,
    int device_id,
    double* xyz_host,
    double* radii_host,
    double* stats_out,
    int stats_len
);

int sgpu_trace_period(
    void* handle,
    const double* R0_host,
    const double* Z0_host,
    double* R1_host,
    double* Z1_host,
    int n_lines,
    int nfp,
    int steps
);

int sgpu_trace_period_blockline(
    void* handle,
    const double* R0_host,
    const double* Z0_host,
    double* R1_host,
    double* Z1_host,
    int n_lines,
    int nfp,
    int steps,
    int threads_per_line
);

int sgpu_trace_period_blockline_mixed(
    void* handle,
    const double* R0_host,
    const double* Z0_host,
    double* R1_host,
    double* Z1_host,
    int n_lines,
    int nfp,
    int steps,
    int threads_per_line,
    int mode
);

int sgpu_trace_axis_samples(
    void* handle,
    double R0,
    double Z0,
    int nfp,
    int integration_steps,
    int n_samples,
    double* R_host,
    double* Z_host,
    double* R_phi_host,
    double* Z_phi_host
);

const char* sgpu_last_error();

}
