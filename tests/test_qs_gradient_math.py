from __future__ import annotations

import numpy as np
import torch


def point_residual_vjp_numpy(b, jacobian, p, iota, helicity, nfp, G, lambdas):
    magnitude = np.linalg.norm(b)
    q = jacobian.T @ b / magnitude
    cross = np.cross(b, p)
    A = cross @ q
    C = b @ q
    M, N = helicity
    values = np.asarray(
        [((M * iota - N) * A - M * G * C), iota * A - G * C, -nfp * A]
    ) / magnitude**3
    lt, lqa, lqp = lambdas
    adj_A = (lt * (M * iota - N) + lqa * iota - lqp * nfp) / magnitude**3
    adj_C = (-lt * M * G - lqa * G) / magnitude**3
    adj_magnitude = -3.0 * np.dot(lambdas, values * magnitude**3) / magnitude**4
    adj_G = (-lt * M * C - lqa * C) / magnitude**3
    adj_b = adj_A * np.cross(p, q) + adj_C * q
    adj_q = adj_A * cross + adj_C * b
    adj_y = adj_q / magnitude
    adj_magnitude -= np.dot(adj_q, q) / magnitude
    adj_b += jacobian @ adj_y + adj_magnitude * b / magnitude
    adj_jacobian = np.outer(b, adj_y)
    return values, adj_b, adj_jacobian, adj_G


def test_point_residual_vjp_matches_autograd() -> None:
    generator = np.random.default_rng(73)
    b_np = generator.normal(size=3)
    jacobian_np = generator.normal(size=(3, 3))
    p_np = generator.normal(size=3)
    lambdas_np = generator.normal(size=3)
    iota = 1.7
    helicity = (1, 6)
    nfp = 6
    G_value = 0.21
    values, adj_b, adj_jacobian, adj_G = point_residual_vjp_numpy(
        b_np, jacobian_np, p_np, iota, helicity, nfp, G_value, lambdas_np
    )

    b = torch.tensor(b_np, dtype=torch.float64, requires_grad=True)
    jacobian = torch.tensor(jacobian_np, dtype=torch.float64, requires_grad=True)
    p = torch.tensor(p_np, dtype=torch.float64)
    G = torch.tensor(G_value, dtype=torch.float64, requires_grad=True)
    magnitude = torch.linalg.vector_norm(b)
    q = jacobian.T @ b / magnitude
    A = torch.dot(torch.linalg.cross(b, p), q)
    C = torch.dot(b, q)
    torch_values = torch.stack(
        [
            ((helicity[0] * iota - helicity[1]) * A - helicity[0] * G * C),
            iota * A - G * C,
            -nfp * A,
        ]
    ) / magnitude**3
    objective = torch.dot(torch_values, torch.tensor(lambdas_np, dtype=torch.float64))
    torch_adj_b, torch_adj_jacobian, torch_adj_G = torch.autograd.grad(
        objective, (b, jacobian, G)
    )
    np.testing.assert_allclose(values, torch_values.detach().numpy(), rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(adj_b, torch_adj_b.numpy(), rtol=2.0e-13, atol=2.0e-13)
    np.testing.assert_allclose(adj_jacobian, torch_adj_jacobian.numpy(), rtol=2.0e-13, atol=2.0e-13)
    np.testing.assert_allclose(adj_G, torch_adj_G.item(), rtol=2.0e-13, atol=2.0e-13)


def segment_vjp_numpy(point, segment_position, weight, adj_b, adj_jacobian):
    mu = 1.0e-7
    r = point - segment_position
    r2 = r @ r
    invr = 1.0 / np.sqrt(r2)
    invr3 = invr**3
    invr5 = invr**5
    invr7 = invr**7
    u = np.cross(weight, r)
    basis = np.eye(3)
    K = np.column_stack([np.cross(weight, basis[:, index]) for index in range(3)])
    adj_u = mu * invr3 * adj_b
    adj_f = mu * (adj_b @ u) + mu * np.sum(adj_jacobian * K)
    adj_u += -3.0 * mu * invr5 * (adj_jacobian @ r)
    adj_r = -3.0 * mu * invr5 * (adj_jacobian.T @ u)
    adj_g = -3.0 * mu * np.sum(adj_jacobian * np.outer(u, r))
    adj_K = mu * invr3 * adj_jacobian
    adj_weight = sum(np.cross(basis[:, index], adj_K[:, index]) for index in range(3))
    adj_weight += np.cross(r, adj_u)
    adj_r += np.cross(adj_u, weight)
    adj_r += (-3.0 * adj_f * invr5 - 5.0 * adj_g * invr7) * r
    return -adj_r, adj_weight


def test_segment_biot_savart_vjp_matches_autograd() -> None:
    generator = np.random.default_rng(91)
    point_np = generator.normal(size=3)
    position_np = generator.normal(size=3)
    weight_np = generator.normal(size=3)
    adj_b_np = generator.normal(size=3)
    adj_jacobian_np = generator.normal(size=(3, 3))
    adj_position, adj_weight = segment_vjp_numpy(
        point_np, position_np, weight_np, adj_b_np, adj_jacobian_np
    )

    point = torch.tensor(point_np, dtype=torch.float64)
    position = torch.tensor(position_np, dtype=torch.float64, requires_grad=True)
    weight = torch.tensor(weight_np, dtype=torch.float64, requires_grad=True)
    r = point - position
    magnitude = torch.linalg.vector_norm(r)
    u = torch.linalg.cross(weight, r)
    basis = torch.eye(3, dtype=torch.float64)
    K = torch.stack([torch.linalg.cross(weight, basis[:, index]) for index in range(3)], dim=1)
    B = 1.0e-7 * u / magnitude**3
    jacobian = 1.0e-7 * (K / magnitude**3 - 3.0 * torch.outer(u, r) / magnitude**5)
    objective = torch.dot(B, torch.tensor(adj_b_np, dtype=torch.float64)) + torch.sum(
        jacobian * torch.tensor(adj_jacobian_np, dtype=torch.float64)
    )
    torch_position, torch_weight = torch.autograd.grad(objective, (position, weight))
    np.testing.assert_allclose(adj_position, torch_position.numpy(), rtol=3.0e-13, atol=3.0e-13)
    np.testing.assert_allclose(adj_weight, torch_weight.numpy(), rtol=3.0e-13, atol=3.0e-13)


def scaled_ridge_ls_vjp_numpy(matrix, rhs, ridge, adj_solution, adj_relative):
    scales = np.linalg.norm(matrix, axis=0)
    scaled_matrix = matrix / scales
    augmented = np.vstack([scaled_matrix, np.sqrt(ridge) * np.eye(matrix.shape[1])])
    augmented_rhs = np.concatenate([rhs, np.zeros(matrix.shape[1])])
    scaled_solution = np.linalg.lstsq(augmented, augmented_rhs, rcond=None)[0]
    solution = scaled_solution / scales
    residual = scaled_matrix @ scaled_solution - rhs
    residual_norm = np.linalg.norm(residual)
    rhs_norm = np.linalg.norm(rhs)

    adj_residual = adj_relative * residual / (residual_norm * rhs_norm)
    adj_rhs = -adj_residual - adj_relative * residual_norm * rhs / rhs_norm**3
    adj_scaled_solution = adj_solution / scales + scaled_matrix.T @ adj_residual
    normal = scaled_matrix.T @ scaled_matrix + ridge * np.eye(matrix.shape[1])
    multiplier = np.linalg.solve(normal, adj_scaled_solution)
    matrix_multiplier = scaled_matrix @ multiplier
    adj_rhs += matrix_multiplier
    adj_scaled_matrix = (
        np.outer(adj_residual, scaled_solution)
        - np.outer(residual, multiplier)
        - np.outer(matrix_multiplier, scaled_solution)
    )
    scale_adjoint = -adj_solution * scaled_solution / scales**2
    scale_adjoint -= np.sum(adj_scaled_matrix * scaled_matrix, axis=0) / scales
    adj_matrix = adj_scaled_matrix / scales + scaled_matrix * scale_adjoint
    return solution, residual_norm / rhs_norm, adj_matrix, adj_rhs


def test_scaled_ridge_ls_vjp_matches_autograd() -> None:
    generator = np.random.default_rng(117)
    matrix_np = generator.normal(size=(19, 7))
    rhs_np = generator.normal(size=19)
    adj_solution_np = generator.normal(size=7)
    adj_relative = -0.37
    ridge = 2.0e-3
    solution, relative, adj_matrix, adj_rhs = scaled_ridge_ls_vjp_numpy(
        matrix_np, rhs_np, ridge, adj_solution_np, adj_relative
    )

    matrix = torch.tensor(matrix_np, dtype=torch.float64, requires_grad=True)
    rhs = torch.tensor(rhs_np, dtype=torch.float64, requires_grad=True)
    scales = torch.linalg.vector_norm(matrix, dim=0)
    scaled_matrix = matrix / scales
    augmented = torch.cat(
        [scaled_matrix, np.sqrt(ridge) * torch.eye(matrix.shape[1], dtype=torch.float64)]
    )
    augmented_rhs = torch.cat([rhs, torch.zeros(matrix.shape[1], dtype=torch.float64)])
    scaled_solution = torch.linalg.lstsq(augmented, augmented_rhs).solution
    torch_solution = scaled_solution / scales
    residual = scaled_matrix @ scaled_solution - rhs
    torch_relative = torch.linalg.vector_norm(residual) / torch.linalg.vector_norm(rhs)
    objective = torch.dot(
        torch_solution, torch.tensor(adj_solution_np, dtype=torch.float64)
    ) + adj_relative * torch_relative
    torch_adj_matrix, torch_adj_rhs = torch.autograd.grad(objective, (matrix, rhs))
    np.testing.assert_allclose(solution, torch_solution.detach().numpy(), rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(relative, torch_relative.item(), rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(adj_matrix, torch_adj_matrix.numpy(), rtol=3.0e-11, atol=3.0e-11)
    np.testing.assert_allclose(adj_rhs, torch_adj_rhs.numpy(), rtol=3.0e-11, atol=3.0e-11)


def test_normalized_alpha_weight_vjp_matches_autograd() -> None:
    generator = np.random.default_rng(131)
    base_np = np.exp(generator.normal(size=23))
    adj_weight_np = generator.normal(size=23)
    count = base_np.size
    scale = np.sqrt(count / np.dot(base_np, base_np))
    weight_np = scale * base_np
    dot_adjoint_weight = np.dot(adj_weight_np, weight_np)
    adj_base = scale * (adj_weight_np - weight_np * dot_adjoint_weight / count)

    base = torch.tensor(base_np, dtype=torch.float64, requires_grad=True)
    weight = np.sqrt(count) * base / torch.linalg.vector_norm(base)
    objective = torch.dot(weight, torch.tensor(adj_weight_np, dtype=torch.float64))
    (torch_adj_base,) = torch.autograd.grad(objective, base)
    np.testing.assert_allclose(adj_base, torch_adj_base.numpy(), rtol=2.0e-13, atol=2.0e-13)
