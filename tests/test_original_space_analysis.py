from pathlib import Path

from scripts.analyze_original_space_optimization import parse_run


def test_parse_run_allows_equals_in_label() -> None:
    label, path = parse_run("data h=.0025 lr=.01=results/run")

    assert label == "data h=.0025 lr=.01"
    assert path == Path("results/run")
