import json
from pathlib import Path


def test_cleaned_files_exist_after_run():
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "cleaned" / "panel_no_covariate_trend.csv",
        root / "cleaned" / "panel_covariate_trend.csv",
    ]
    for path in required:
        assert path.exists(), f"Missing file: {path}"


def test_output_tables_exist_after_run():
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "output" / "group_summary_no_covariate_trend.csv",
        root / "output" / "event_study_no_covariate_trend.csv",
        root / "output" / "event_study_covariate_trend_never.csv",
        root / "output" / "event_study_covariate_trend_notyet.csv",
    ]
    for path in required:
        assert path.exists(), f"Missing file: {path}"


def test_results_file_exists_after_run():
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "output" / "results.json",
    ]
    for path in required:
        assert path.exists(), f"Missing file: {path}"


def test_results_json_twfe_keys():
    root = Path(__file__).resolve().parents[1]
    results = json.loads((root / "output" / "results.json").read_text(encoding="utf-8"))
    required = {
        "twfe_no_covariate_trend",
        "twfe_covariate_trend",
    }
    assert required.issubset(results.keys())


def test_results_json_event_study_keys():
    root = Path(__file__).resolve().parents[1]
    results = json.loads((root / "output" / "results.json").read_text(encoding="utf-8"))
    required = {
        "event_study_no_covariate_simple_att",
        "event_study_covariate_never_simple_att",
        "event_study_covariate_notyet_simple_att",
    }
    assert required.issubset(results.keys())


def test_results_json_shape_keys():
    root = Path(__file__).resolve().parents[1]
    results = json.loads((root / "output" / "results.json").read_text(encoding="utf-8"))
    required = {
        "group_summary_rows",
        "event_study_rows",
    }
    assert required.issubset(results.keys())


def test_rendered_html_exists():
    root = Path(__file__).resolve().parents[1]
    html_path = root / "report" / "solution.html"
    assert html_path.exists(), "Missing report/solution.html."


def test_rendered_html_contains_required_sections():
    root = Path(__file__).resolve().parents[1]
    html_path = root / "report" / "solution.html"
    html = html_path.read_text(encoding="utf-8")
    required_strings = [
        "Group shares and treatment effects",
        "Event-study DID estimates",
        "TWFE comparison",
    ]
    for token in required_strings:
        assert token in html


def test_rendered_html_contains_result_key():
    root = Path(__file__).resolve().parents[1]
    html_path = root / "report" / "solution.html"
    html = html_path.read_text(encoding="utf-8")
    required_strings = [
        "twfe_no_covariate_trend",
    ]
    for token in required_strings:
        assert token in html
