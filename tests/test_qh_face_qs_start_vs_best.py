from scripts.plot_qh_face_qs_start_vs_best import build_pairs, pair_summary


def row(trajectory: str, iteration: int, qh: float, accepted: bool = True) -> dict[str, str]:
    return {
        "trajectory_id": trajectory,
        "iteration": str(iteration),
        "surface_name": "fixed_probe",
        "face_qh": str(qh),
        "accepted": str(accepted),
        "nfp": "4",
        "n_base_coils": "3",
    }


def test_build_pairs_keeps_endpoint_and_best_measured_later_distinct() -> None:
    pairs = build_pairs(
        [
            row("a", 0, 1.0e-3),
            row("a", 10, 4.0e-4),
            row("a", 200, 6.0e-4),
            row("b", 0, 2.0e-3),
            row("b", 10, 1.0e-3, accepted=False),
            row("b", 200, 3.0e-3),
        ]
    )

    assert pairs[0]["endpoint_face_qh"] == 6.0e-4
    assert pairs[0]["best_observed_later_face_qh"] == 4.0e-4
    assert pairs[0]["best_observed_later_iteration"] == "10"
    assert pairs[1]["best_observed_later_face_qh"] == 3.0e-3
    assert pair_summary(pairs, "endpoint_face_qh")["improved_count"] == 1
