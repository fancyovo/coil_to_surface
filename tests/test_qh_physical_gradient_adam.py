from scripts.optimize_qh_physical_gradient_adam import accept_exact_score_candidate


def test_exact_score_gate_requires_ok_and_non_decreasing_score() -> None:
    current = {"status": "ok", "score": 60.0}

    assert accept_exact_score_candidate(
        current, {"status": "ok", "score": 60.0}, accept_drop=0.0
    )
    assert accept_exact_score_candidate(
        current, {"status": "ok", "score": 60.1}, accept_drop=0.0
    )
    assert not accept_exact_score_candidate(
        current, {"status": "ok", "score": 59.999}, accept_drop=0.0
    )
    assert accept_exact_score_candidate(
        current, {"status": "ok", "score": 59.95}, accept_drop=0.05
    )
    assert not accept_exact_score_candidate(
        current, {"status": "no_axis", "score": 80.0}, accept_drop=100.0
    )
