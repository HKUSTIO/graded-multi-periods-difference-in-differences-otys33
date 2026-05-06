import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.did_multiperiod import (
    aggregate_post_treatment_effects,
    estimate_event_study,
    estimate_twfe_coefficient,
    summarize_group_shares_and_att,
)

import pandas as pd


def main() -> None:
    config = json.loads((ROOT / "config" / "assignment.json").read_text(encoding="utf-8"))
    cleaned_dir = ROOT / "cleaned"
    output_dir = ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    event_times = [int(x) for x in config["event_times"]]
    panel_no_cov = pd.read_csv(cleaned_dir / "panel_no_covariate_trend.csv")
    panel_cov = pd.read_csv(cleaned_dir / "panel_covariate_trend.csv")

    summary_no_cov = summarize_group_shares_and_att(panel_no_cov)
    event_study_no_cov = estimate_event_study(panel_no_cov, event_times, control_group="never")
    event_study_cov_never = estimate_event_study(panel_cov, event_times, control_group="never")
    event_study_cov_notyet = estimate_event_study(panel_cov, event_times, control_group="notyet")

    summary_no_cov.to_csv(output_dir / "group_summary_no_covariate_trend.csv", index=False)
    event_study_no_cov.to_csv(output_dir / "event_study_no_covariate_trend.csv", index=False)
    event_study_cov_never.to_csv(output_dir / "event_study_covariate_trend_never.csv", index=False)
    event_study_cov_notyet.to_csv(output_dir / "event_study_covariate_trend_notyet.csv", index=False)

    results = {
        "twfe_no_covariate_trend": estimate_twfe_coefficient(panel_no_cov),
        "twfe_covariate_trend": estimate_twfe_coefficient(panel_cov),
        "event_study_no_covariate_simple_att": aggregate_post_treatment_effects(event_study_no_cov),
        "event_study_covariate_never_simple_att": aggregate_post_treatment_effects(event_study_cov_never),
        "event_study_covariate_notyet_simple_att": aggregate_post_treatment_effects(event_study_cov_notyet),
        "group_summary_rows": int(summary_no_cov.shape[0]),
        "event_study_rows": int(event_study_no_cov.shape[0]),
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
