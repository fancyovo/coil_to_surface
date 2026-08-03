from copy import deepcopy

import numpy as np

from stellarator_eval.volume_score import evaluate_volume_quality_score


def _result(
    *, inverse_aspect=0.05, qs_error=0.02, iota=1.2, target=(1, 3),
    qa_error=0.2, qp_error=0.6,
):
    radius = inverse_aspect
    return {
        "status": "ok",
        "axis": {
            "has_axis": True,
            "best_R": 1.0,
            "best_residual": 1e-7,
            "topology_class": "elliptic",
            "topology_ellipse_aspect": 1.2,
        },
        "psi": {
            "fit_info": {
                "validation_angle_p95": 2e-4,
                "validation_angle_l2": 8e-5,
                "train_rms": 5e-4,
            }
        },
        "surface_screen": {
            "levels": [
                {
                    "psi_level": 0.49,
                    "ok": True,
                    "radius_mean": radius,
                    "rel_end_distance_p95": 0.005,
                }
            ],
            "stable_accepted_levels": [0.49],
            "volume_qs_accepted_levels": [0.49],
        },
        "volume_qs": {
            "status": "ok",
            "target_helicity": list(target),
            "s_edge": 0.49,
            "flux": {
                "diagnostics": {
                    "section_relative_std_edge": 1e-3,
                    "boundary_residual_max": 1e-10,
                    "monotone": True,
                    "boundary_effective_minor_radius_edge": radius,
                    "boundary_axis_major_radius_mean": 1.0,
                    "boundary_effective_inverse_aspect_ratio": inverse_aspect,
                    "boundary_volume_edge": 2.0 * np.pi**2 * radius**2,
                }
            },
            "alpha": {
                "diagnostics": {
                    "relative_l2": 0.1,
                    "normal_B_relative_l2": 2e-5,
                    "iota_min": iota,
                    "iota_max": iota,
                }
            },
            "metrics": {
                "target": {
                    "f_C_over_B3_rms": qs_error,
                    "radial_bins": [{"f_C_over_B3_rms": 1.2 * qs_error}],
                },
                "QA": {"f_C_over_B3_rms": qa_error},
                "QP": {"f_C_over_B3_rms": qp_error},
            },
        },
    }


def test_volume_score_is_finite_and_reports_stage():
    score = evaluate_volume_quality_score(_result())
    assert 0.0 <= score["score"] <= 100.0
    assert score["status"] == "volume_qs"
    assert set(score["components"]) == {
        "axis",
        "psi",
        "surface",
        "coordinate",
        "volume_qs",
        "iota",
        "coil",
    }


def test_useful_qs_rewards_larger_surface_at_fixed_residual():
    small = evaluate_volume_quality_score(_result(inverse_aspect=0.01))
    large = evaluate_volume_quality_score(_result(inverse_aspect=0.06))
    assert large["components"]["surface"] > small["components"]["surface"]
    assert large["components"]["volume_qs"] > small["components"]["volume_qs"]
    assert large["score"] > small["score"]


def test_surface_size_reward_saturates_when_surface_is_large_enough():
    enough = evaluate_volume_quality_score(_result(inverse_aspect=0.03))
    huge = evaluate_volume_quality_score(_result(inverse_aspect=0.12))
    assert enough["components"]["surface"] == huge["components"]["surface"]
    assert enough["components"]["volume_qs"] == huge["components"]["volume_qs"]


def test_qh_iota_below_one_is_penalized_monotonically():
    very_low = evaluate_volume_quality_score(_result(iota=0.1))
    low = evaluate_volume_quality_score(_result(iota=0.7))
    enough = evaluate_volume_quality_score(_result(iota=1.0))
    high = evaluate_volume_quality_score(_result(iota=1.4))
    assert very_low["components"]["iota"] < low["components"]["iota"]
    assert low["components"]["iota"] < enough["components"]["iota"]
    assert enough["components"]["iota"] == high["components"]["iota"] == 100.0
    assert very_low["components"]["volume_qs"] < low["components"]["volume_qs"]
    assert very_low["score"] < low["score"] < enough["score"]
    assert np.isclose(very_low["details"]["score_qh_total_iota_factor"], 0.109)
    assert enough["details"]["score_qh_total_iota_factor"] == 1.0


def test_qa_does_not_apply_qh_iota_gate():
    qa = evaluate_volume_quality_score(_result(iota=0.1, target=(1, 0)))
    assert qa["components"]["iota"] == 100.0
    assert qa["details"]["volume_qs_iota_factor"] == 1.0
    assert qa["details"]["score_qh_total_iota_factor"] == 1.0


def test_qp_does_not_apply_qh_iota_gate():
    qp = evaluate_volume_quality_score(_result(iota=0.1, target=(0, 3)))
    assert qp["components"]["iota"] == 100.0
    assert qp["details"]["volume_qs_iota_factor"] == 1.0
    assert qp["details"]["score_qh_total_iota_factor"] == 1.0


def test_volume_score_penalizes_worse_qs_at_fixed_surface():
    good_result = _result(qs_error=0.005)
    bad_result = deepcopy(good_result)
    bad_result["volume_qs"]["metrics"]["target"]["f_C_over_B3_rms"] = 0.3
    bad_result["volume_qs"]["metrics"]["target"]["radial_bins"][-1][
        "f_C_over_B3_rms"
    ] = 0.36
    good = evaluate_volume_quality_score(good_result)
    bad = evaluate_volume_quality_score(bad_result)
    assert good["components"]["volume_qs"] > bad["components"]["volume_qs"]
    assert good["score"] > bad["score"]


def test_qh_target_must_outperform_qa_and_qp():
    qh_dominant = evaluate_volume_quality_score(
        _result(qs_error=0.01, qa_error=0.2, qp_error=0.6)
    )
    qp_dominant = evaluate_volume_quality_score(
        _result(qs_error=0.2, qa_error=0.02, qp_error=0.03)
    )
    assert qh_dominant["details"]["qh_helicity_advantage"] > 0.9
    assert qp_dominant["details"]["qh_helicity_advantage"] < 0.2
    assert qh_dominant["details"]["score_qh_helicity_quality"] == 1.0
    assert qp_dominant["details"]["score_qh_helicity_quality"] < 0.5
    assert qh_dominant["score"] > qp_dominant["score"]


def test_failure_statuses_remain_finite():
    no_axis = {"status": "failed", "axis": {"has_axis": False}}
    score = evaluate_volume_quality_score(no_axis)
    assert score["status"] == "no_axis"
    assert np.isfinite(score["score"])
