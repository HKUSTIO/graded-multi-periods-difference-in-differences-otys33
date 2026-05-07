from __future__ import annotations

import math

import numpy as np
import pandas as pd


def logistic(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(x))))


def generate_panel_data(config: dict, heterogeneous_trend: bool) -> pd.DataFrame:
    rng = np.random.default_rng(int(config["seed_population"]))
    n_units = int(config["n_units"])
    t0 = int(config["pre_periods"])
    t_total = t0 + int(config["post_periods"])
    times = np.arange(1, t_total + 1)

    x_draw = rng.uniform(0.0, 1.0, size=n_units)
    x1 = (x_draw >= 0.3).astype(int)
    x2 = (x_draw >= 0.7).astype(int)
    u = rng.uniform(0.0, 1.0, size=n_units)

    alpha0 = float(config["adoption_intercept"])
    alpha1 = float(config["adoption_slope_middle"])
    alpha2 = float(config["adoption_slope_late"])
    p1 = logistic(alpha0)
    p2 = np.array([logistic(alpha0 + alpha1 * value) for value in x1])
    p3 = np.array([logistic(alpha0 + alpha1 * value) for value in x1 + x2])
    p4 = np.array([logistic(alpha0 + alpha2 * value) for value in x1 + x2])

    cohort = np.zeros(n_units, dtype=int)
    cohort[u <= p1] = t0 + 1
    cohort[(u > p1) & (u <= p2)] = t0 + 2
    cohort[(u > p2) & (u <= p3)] = t0 + 3
    cohort[(u > p3) & (u <= p4)] = t0 + 4

    individual_effect = rng.normal(0.0, float(config["individual_sd"]), size=n_units)
    tau_i = rng.normal(float(config["tau_mean"]), float(config["tau_sd"]), size=n_units)
    time_shocks = rng.uniform(float(config["tau_time_low"]), float(config["tau_time_high"]), size=t_total + 1)

    rows = []
    for time in times:
        if heterogeneous_trend:
            trend_cfg = config["heterogeneous_trend"]
            trend = (
                (time / t_total) * float(trend_cfg["baseline_slope"]) * (1 - x1 - x2)
                + (time / t_total) * float(trend_cfg["x1_slope"]) * x1
                + (time / t_total) * float(trend_cfg["x2_slope"]) * x2
            )
        else:
            trend = np.full(n_units, time / t_total)

        error = rng.normal(0.0, float(config["error_sd"]), size=n_units)
        treated_ever = (cohort > 0).astype(int)
        y0 = float(config["base_level"]) + treated_ever * (-individual_effect) + (1 - treated_ever) * individual_effect + trend + error

        d = ((cohort > 0) & (time >= cohort)).astype(int)
        multiplier = np.zeros(n_units)
        multiplier[cohort == t0 + 1] = 1.0
        multiplier[cohort == t0 + 2] = -2.5
        multiplier[cohort == t0 + 3] = -1.75
        multiplier[cohort == t0 + 4] = -1.0
        tau_it = time_shocks[time] * np.abs(tau_i) * multiplier
        y = y0 + d * tau_it
        relative_time = np.where(cohort > 0, time - cohort, 0)

        for idx in range(n_units):
            rows.append(
                {
                    "id": idx + 1,
                    "x1": int(x1[idx]),
                    "x2": int(x2[idx]),
                    "cohort": int(cohort[idx]),
                    "time": int(time),
                    "relative_time": int(relative_time[idx]),
                    "d": int(d[idx]),
                    "y0": float(y0[idx]),
                    "tau_it": float(tau_it[idx]),
                    "y": float(y[idx]),
                }
            )

    return pd.DataFrame(rows, columns=["id", "x1", "x2", "cohort", "time", "relative_time", "d", "y0", "tau_it", "y"])


def summarize_group_shares_and_att(data: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per treated cohort and one row for all treated observations.
    """
    units = data[["id", "cohort"]].drop_duplicates()
    n_units = units.shape[0]
    rows = []

    for cohort in sorted(units.loc[units["cohort"] > 0, "cohort"].unique()):
        cohort_rows = data[(data["cohort"] == cohort) & (data["d"] == 1)]
        rows.append(
            {
                "group": f"cohort_{int(cohort)}",
                "fraction": float((units["cohort"] == cohort).sum() / n_units),
                "att": float(cohort_rows["tau_it"].mean()),
            }
        )

    treated_rows = data[data["d"] == 1]
    rows.append(
        {
            "group": "all_treated",
            "fraction": float(treated_rows.shape[0] / data.shape[0]),
            "att": float(treated_rows["tau_it"].mean()),
        }
    )

    return pd.DataFrame(rows, columns=["group", "fraction", "att"])


def estimate_cohort_did(data: pd.DataFrame, cohort: int, event_time: int, control_group: str) -> float:
    """
    Return a two-period DID estimate for one treatment cohort and event time.
    """
    target_time = cohort + event_time
    baseline_time = cohort - 1
    treated = data["cohort"] == cohort

    if control_group == "never":
        controls = data["cohort"] == 0
    elif control_group == "notyet":
        controls = (data["cohort"] == 0) | (data["cohort"] > target_time)
    else:
        raise ValueError("control_group must be 'never' or 'notyet'.")

    treated_target = data.loc[treated & (data["time"] == target_time), "y"].mean()
    treated_baseline = data.loc[treated & (data["time"] == baseline_time), "y"].mean()
    control_target = data.loc[controls & (data["time"] == target_time), "y"].mean()
    control_baseline = data.loc[controls & (data["time"] == baseline_time), "y"].mean()

    return float((treated_target - treated_baseline) - (control_target - control_baseline))


def estimate_event_study(data: pd.DataFrame, event_times: list[int], control_group: str) -> pd.DataFrame:
    """
    Return cohort-event DID estimates.
    """
    min_time = data["time"].min()
    max_time = data["time"].max()
    cohorts = sorted(data.loc[data["cohort"] > 0, "cohort"].unique())
    rows = []

    for cohort in cohorts:
        for event_time in sorted(event_times):
            target_time = int(cohort) + int(event_time)
            baseline_time = int(cohort) - 1
            if min_time <= baseline_time <= max_time and min_time <= target_time <= max_time:
                rows.append(
                    {
                        "cohort": int(cohort),
                        "event_time": int(event_time),
                        "estimate": estimate_cohort_did(data, int(cohort), int(event_time), control_group),
                    }
                )

    return pd.DataFrame(rows, columns=["cohort", "event_time", "estimate"]).sort_values(
        ["cohort", "event_time"], ignore_index=True
    )


def aggregate_post_treatment_effects(event_study: pd.DataFrame) -> float:
    """
    Return the average estimate over post-treatment event times.
    """
    post = event_study.loc[event_study["event_time"] >= 0, "estimate"]
    return float(post.mean())


def estimate_twfe_coefficient(data: pd.DataFrame) -> float:
    """
    Return the coefficient from a residualized two-way fixed effects regression of y on d.
    """
    y = data["y"]
    d = data["d"]

    y_residual = y - data.groupby("id")["y"].transform("mean") - data.groupby("time")["y"].transform("mean") + y.mean()
    d_residual = d - data.groupby("id")["d"].transform("mean") - data.groupby("time")["d"].transform("mean") + d.mean()

    return float((d_residual * y_residual).sum() / (d_residual**2).sum())
