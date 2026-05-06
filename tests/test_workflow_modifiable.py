from pathlib import Path


def test_required_top_level_files_exist():
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "README.md",
        root / "pyproject.toml",
    ]
    for path in required:
        assert path.exists(), f"Missing file: {path}"


def test_required_config_and_source_files_exist():
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "config" / "assignment.json",
        root / "src" / "did_multiperiod.py",
    ]
    for path in required:
        assert path.exists(), f"Missing file: {path}"


def test_required_script_files_exist():
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "scripts" / "run_cleaning.py",
        root / "scripts" / "run_analysis.py",
        root / "scripts" / "run_pipeline.py",
        root / "scripts" / "run_assignment.py",
    ]
    for path in required:
        assert path.exists(), f"Missing file: {path}"


def test_required_report_file_exists():
    root = Path(__file__).resolve().parents[1]
    required = [
        root / "report" / "solution.qmd",
    ]
    for path in required:
        assert path.exists(), f"Missing file: {path}"


def test_report_references_run_script():
    root = Path(__file__).resolve().parents[1]
    report = (root / "report" / "solution.qmd").read_text(encoding="utf-8")
    assert "scripts/run_pipeline.py" in report
