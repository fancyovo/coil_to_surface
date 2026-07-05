#pragma once

#include <cstddef>

extern "C" {

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
    int precision_mode,
    double* coeff_host,
    double* train_rms_out,
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

const char* sgpu_last_error();

}
