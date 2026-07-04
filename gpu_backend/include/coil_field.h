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

const char* sgpu_last_error();

}
