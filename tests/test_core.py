import numpy as np
import pandas as pd

from src.did_multiperiod import (
    aggregate_post_treatment_effects,
    estimate_cohort_did,
    estimate_event_study,
    estimate_twfe_coefficient,
    generate_panel_data,
    logistic,
    summarize_group_shares_and_att,
)


def test_logistic_matches_reference_values():
    assert np.isclose(logistic(0.0), 0.5)
    assert np.isclose(logistic(-0.25), np.exp(-0.25) / (1.0 + np.exp(-0.25)))


def test_generate_panel_data_columns_and_timing():
    config = {
        "seed_population": 11,
        "n_units": 80,
        "pre_periods": 4,
        "post_periods": 6,
        "base_level": 5.4,
        "individual_sd": 0.15,
        "error_sd": 0.03,
        "tau_mean": 0.1,
        "tau_sd": 0.2,
        "tau_time_low": 0.9,
        "tau_time_high": 1.1,
        "adoption_intercept": -0.25,
        "adoption_slope_middle": 0.75,
        "adoption_slope_late": 1.0,
        "heterogeneous_trend": {"baseline_slope": 1.0, "x1_slope": -0.2, "x2_slope": -0.1},
    }
    data = generate_panel_data(config, heterogeneous_trend=False)
    assert list(data.columns) == ["id", "x1", "x2", "cohort", "time", "relative_time", "d", "y0", "tau_it", "y"]
    assert data.shape[0] == 800
    treated_rows = data[data["cohort"] > 0]
    assert (treated_rows["d"] == (treated_rows["time"] >= treated_rows["cohort"]).astype(int)).all()


def make_summary_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3, 4, 4],
            "cohort": [3, 3, 3, 3, 4, 4, 0, 0],
            "time": [2, 3, 2, 3, 3, 4, 2, 3],
            "d": [0, 1, 0, 1, 0, 1, 0, 0],
            "tau_it": [0.0, 2.0, 0.0, 4.0, 0.0, -1.0, 0.0, 0.0],
        }
    )


def test_summarize_group_shares_columns():
    summary = summarize_group_shares_and_att(make_summary_panel())
    assert list(summary.columns) == ["group", "fraction", "att"]


def test_summarize_group_shares_rows():
    summary = summarize_group_shares_and_att(make_summary_panel())
    assert set(summary["group"]) == {"cohort_3", "cohort_4", "all_treated"}


def test_summarize_group_att_values():
    summary = summarize_group_shares_and_att(make_summary_panel())
    assert np.isclose(summary.loc[summary["group"] == "cohort_3", "att"].iloc[0], 3.0)


def test_estimate_cohort_did_known_panel():
    data = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3, 4, 4],
            "cohort": [3, 3, 3, 3, 0, 0, 0, 0],
            "time": [2, 3, 2, 3, 2, 3, 2, 3],
            "d": [0, 1, 0, 1, 0, 0, 0, 0],
            "y": [10.0, 15.0, 12.0, 17.0, 8.0, 9.0, 9.0, 10.0],
        }
    )
    estimate = estimate_cohort_did(data, cohort=3, event_time=0, control_group="never")
    assert np.isclose(estimate, 4.0)


def test_estimate_cohort_did_notyet_controls():
    data = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3, 4, 4],
            "cohort": [3, 3, 4, 4, 0, 0, 0, 0],
            "time": [2, 3, 2, 3, 2, 3, 2, 3],
            "d": [0, 1, 0, 0, 0, 0, 0, 0],
            "y": [10.0, 15.0, 6.0, 7.0, 8.0, 9.0, 9.0, 10.0],
        }
    )
    estimate = estimate_cohort_did(data, cohort=3, event_time=0, control_group="notyet")
    assert np.isclose(estimate, 4.0)


def make_event_study_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "cohort": [3, 3, 3, 3, 3, 3, 0, 0, 0],
            "time": [2, 3, 4, 2, 3, 4, 2, 3, 4],
            "d": [0, 1, 1, 0, 1, 1, 0, 0, 0],
            "y": [1.0, 4.0, 5.0, 2.0, 5.0, 6.0, 1.0, 2.0, 3.0],
        }
    )


def test_event_study_columns():
    event_study = estimate_event_study(make_event_study_panel(), event_times=[0, 1], control_group="never")
    assert list(event_study.columns) == ["cohort", "event_time", "estimate"]


def test_event_study_estimates():
    event_study = estimate_event_study(make_event_study_panel(), event_times=[0, 1], control_group="never")
    assert np.allclose(event_study["estimate"].to_numpy(), np.array([2.0, 2.0]))


def test_aggregate_post_treatment_effects():
    event_study = pd.DataFrame(
        {
            "cohort": [3, 3, 4],
            "event_time": [-1, 0, 1],
            "estimate": [10.0, 2.0, 4.0],
        }
    )
    assert np.isclose(aggregate_post_treatment_effects(event_study), 3.0)


def test_twfe_coefficient_recovers_residualized_signal():
    rows = []
    for i, unit_fe in [(1, 0.5), (2, -0.5), (3, 1.0), (4, -1.0)]:
        cohort = 3 if i in [1, 2] else 0
        for time, time_fe in [(1, 0.0), (2, 0.2), (3, 0.4), (4, 0.6)]:
            d = int(cohort == 3 and time >= 3)
            rows.append({"id": i, "time": time, "cohort": cohort, "d": d, "y": unit_fe + time_fe + 2.0 * d})
    data = pd.DataFrame(rows)
    assert np.isclose(estimate_twfe_coefficient(data), 2.0)
