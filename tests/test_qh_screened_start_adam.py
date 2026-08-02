from __future__ import annotations

from scripts.analyze_qh_screened_start_adam import analyze
from scripts.select_qh_screened_adam_start import select_best_row
from scripts.summarize_qh_screened_start_adam import build_summary


def test_select_best_row_is_deterministic_for_ties() -> None:
    rows = [
        {"case_id": 3, "score": 20.0},
        {"case_id": 1, "score": 30.0},
        {"case_id": 2, "score": 30.0},
    ]
    assert select_best_row(rows)["case_id"] == 1


def test_build_summary_checks_thresholds_and_exact_timing() -> None:
    selection = {
        "candidate_seed": 7,
        "candidate_count": 128,
        "nfp": 4,
        "n_base_coils": 3,
        "selected_case_id": 11,
        "selected_status": "ok",
        "selected_score": 35.0,
    }
    adam = {
        "initial_score": 35.00001,
        "final_score": 48.0,
        "best_score": 51.0,
        "best_iteration": 43,
        "completed_iterations": 50,
        "completed_adam_steps": 47,
        "stop_reason": "completed_iterations",
        "total_wall_s": 90.0,
    }
    summary = build_summary(
        selection,
        adam,
        run_started_ns=1_000_000_000,
        selection_finished_ns=4_000_000_000,
        adam_started_ns=5_000_000_000,
        run_finished_ns=95_000_000_000,
        optimizer_seed=8,
    )
    assert summary["crossed_40"]
    assert summary["crossed_50"]
    assert abs(summary["selection_rescore_discrepancy"] - 1.0e-5) < 1.0e-12
    assert summary["timing_s"]["candidate_selection"] == 3.0
    assert summary["timing_s"]["end_to_end"] == 94.0


def test_analyze_reports_multiseed_success_rates() -> None:
    rows = []
    for initial, best, elapsed in ((30.0, 39.0, 100.0), (35.0, 45.0, 110.0), (40.0, 55.0, 120.0)):
        rows.append(
            {
                "initial_score": initial,
                "best_score": best,
                "nfp": 4,
                "n_base_coils": 3,
                "candidate_count": 128,
                "timing_s": {"candidate_selection": 10.0, "end_to_end": elapsed},
            }
        )
    summary = analyze(rows)
    assert summary["thresholds"]["best_ge_40_count"] == 2
    assert summary["thresholds"]["best_ge_50_count"] == 1
    assert summary["timing_s"]["end_to_end_median"] == 110.0
