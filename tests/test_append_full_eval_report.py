import json

from scripts.append_full_eval_to_rl_report import append


def test_append_report_records_selected_sample_and_rejects_duplicate(tmp_path):
    report = tmp_path / "report.md"
    report.write_text("# Report\n", encoding="utf-8")
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(json.dumps({"status": "completed", "source_psi_selection": {"selected_a_m": 0.04},
        "surface_selection": {"selected": {"target_s": 0.49, "final_abs_volume_m3": 1.2, "acceptance_checks": {}}},
        "downstream": {"status": "completed", "surface": {"iota": 1.1, "G": 2, "rho": 1}, "desc": {}},
        "artifacts": {"source_psi_selection": "source.json", "desc": "desc", "evaluation_summary": "summary"}}))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"selection_boundary_next_round": 3, "source_sample_id": "r0002_r00_i001",
        "source_round": 2, "source_rank": 0, "source_sample_index": 1, "source_score": 80,
        "source_components": {"coil": 70}, "case_file": "case.json"}))
    append(report, evaluation, selection)
    assert "r0002_r00_i001" in report.read_text(encoding="utf-8")
    try:
        append(report, evaluation, selection)
    except FileExistsError:
        pass
    else:
        raise AssertionError("duplicate report append was accepted")
