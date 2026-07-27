#pragma once

extern "C" {

int sgpu_internal_eval_B_f32_device(
    void* handle,
    const float* xyz_device,
    float* B_device,
    int n_points
);

int sgpu_internal_eval_B_grad_f32_device(
    void* handle,
    const float* xyz_device,
    float* B_device,
    float* grad_B_device,
    int n_points
);

void sgpu_internal_set_error(const char* message);

}
