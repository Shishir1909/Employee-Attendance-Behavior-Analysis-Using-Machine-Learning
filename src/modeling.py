from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

BASE_FEATURE_COLUMNS = [
    "login_minutes",
    "logout_minutes",
    "total_work_minutes",
    "login_deviation_from_9am",
    "is_weekend",
]


def build_model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build numeric model matrix including one-hot encoded department."""
    matrix = df[BASE_FEATURE_COLUMNS + ["department"]].copy()
    matrix = pd.get_dummies(matrix, columns=["department"], prefix="department")
    return matrix


def _build_global_reason_thresholds(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "short_duration_threshold": float(df["total_work_minutes"].quantile(0.10)),
        "early_logout_threshold": float(df["logout_minutes"].quantile(0.10)),
        "deviation_threshold": float(
            df["login_deviation_from_9am"].abs().quantile(0.90)
        ),
    }


def _reason_for_row_global(row: pd.Series, thresholds: Dict[str, float]) -> str:
    if not row["is_anomaly"]:
        return "normal"

    short_duration = row["total_work_minutes"] <= thresholds["short_duration_threshold"]
    early_logout = row["logout_minutes"] <= thresholds["early_logout_threshold"]
    high_deviation = (
        abs(row["login_deviation_from_9am"]) >= thresholds["deviation_threshold"]
    )

    if (short_duration and early_logout) or (short_duration and high_deviation):
        return "combined_deviation"
    if short_duration:
        return "short_duration"
    if early_logout:
        return "early_logout"
    return "combined_deviation"


def _robust_scale(history: pd.Series, floor: float) -> float:
    q1 = float(history.quantile(0.25))
    q3 = float(history.quantile(0.75))
    iqr = q3 - q1
    robust_std = iqr / 1.349 if iqr > 1e-9 else np.nan
    std = float(history.std(ddof=0))

    if np.isfinite(robust_std) and robust_std > 1e-9:
        return max(robust_std, floor)
    if np.isfinite(std) and std > 1e-9:
        return max(std, floor)
    return floor


def _employee_reason(
    z_login: float,
    z_logout: float,
    z_duration: float,
    threshold: float,
) -> str:
    pattern_scores = {
        "late_login_pattern": max(z_login, 0.0),
        "early_logout_pattern": max(-z_logout, 0.0),
        "short_duration_pattern": max(-z_duration, 0.0),
    }
    best_reason = max(pattern_scores, key=pattern_scores.get)
    if pattern_scores[best_reason] >= threshold:
        return best_reason
    return "combined_deviation_emp"


def _compute_employee_pattern_signals(
    df: pd.DataFrame,
    window_days: int = 7,
    min_history: int = 5,
    threshold: float = 2.0,
) -> pd.DataFrame:
    work_df = df.copy()
    work_df["date"] = pd.to_datetime(work_df["date"], errors="coerce")
    work_df["_orig_order"] = np.arange(len(work_df))
    work_df = work_df.sort_values(by=["employee_id", "date", "login_minutes"]).reset_index(
        drop=True
    )

    login_dev_values = []
    logout_dev_values = []
    duration_dev_values = []
    history_count_values = []
    anomaly_employee_values = []
    reason_employee_values = []
    severity_employee_values = []

    for _, row in work_df.iterrows():
        current_date = row["date"]
        employee_mask = work_df["employee_id"] == row["employee_id"]
        history_mask = (
            employee_mask
            & (work_df["date"] < current_date)
            & (work_df["date"] >= current_date - pd.Timedelta(days=window_days))
        )
        history_df = work_df.loc[history_mask]

        history_count = len(history_df)
        history_count_values.append(history_count)

        if history_count < min_history:
            login_dev_values.append(np.nan)
            logout_dev_values.append(np.nan)
            duration_dev_values.append(np.nan)
            anomaly_employee_values.append(False)
            reason_employee_values.append("insufficient_history")
            severity_employee_values.append(0.0)
            continue

        login_median = float(history_df["login_minutes"].median())
        logout_median = float(history_df["logout_minutes"].median())
        duration_median = float(history_df["total_work_minutes"].median())

        login_dev = float(row["login_minutes"] - login_median)
        logout_dev = float(row["logout_minutes"] - logout_median)
        duration_dev = float(row["total_work_minutes"] - duration_median)

        login_dev_values.append(login_dev)
        logout_dev_values.append(logout_dev)
        duration_dev_values.append(duration_dev)

        login_scale = _robust_scale(history_df["login_minutes"], floor=15.0)
        logout_scale = _robust_scale(history_df["logout_minutes"], floor=15.0)
        duration_scale = _robust_scale(history_df["total_work_minutes"], floor=20.0)

        z_login = login_dev / login_scale
        z_logout = logout_dev / logout_scale
        z_duration = duration_dev / duration_scale

        severity = float(max(abs(z_login), abs(z_logout), abs(z_duration)))
        is_emp_anomaly = severity >= threshold

        anomaly_employee_values.append(is_emp_anomaly)
        severity_employee_values.append(severity)

        if is_emp_anomaly:
            reason_employee_values.append(
                _employee_reason(
                    z_login=z_login,
                    z_logout=z_logout,
                    z_duration=z_duration,
                    threshold=threshold,
                )
            )
        else:
            reason_employee_values.append("normal")

    work_df["login_dev_emp"] = login_dev_values
    work_df["logout_dev_emp"] = logout_dev_values
    work_df["duration_dev_emp"] = duration_dev_values
    work_df["employee_history_count"] = history_count_values
    work_df["is_anomaly_employee"] = anomaly_employee_values
    work_df["reason_employee"] = reason_employee_values
    work_df["severity_employee"] = severity_employee_values

    work_df = work_df.sort_values("_orig_order").drop(columns=["_orig_order"])
    return work_df


def _combine_reason(row: pd.Series) -> str:
    if row["is_anomaly_employee"] and row["is_anomaly"]:
        return "employee_pattern_plus_global_outlier"
    if row["is_anomaly_employee"]:
        return row["reason_employee"]
    if row["is_anomaly"]:
        return row["reason_global"]
    if row["reason_employee"] == "insufficient_history":
        return "insufficient_history"
    return "normal"


def _normalize_series(values: pd.Series) -> pd.Series:
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value - min_value <= 1e-9:
        return pd.Series(0.0, index=values.index)
    return (values - min_value) / (max_value - min_value)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def evaluate_with_ground_truth(scored_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build evaluation summary and confusion matrix if ground truth exists."""
    if "ground_truth_anomaly" not in scored_df.columns:
        return pd.DataFrame(), pd.DataFrame()

    gt_numeric = pd.to_numeric(scored_df["ground_truth_anomaly"], errors="coerce").fillna(0)
    y_true = gt_numeric.astype(int).astype(bool)

    methods = {
        "global": scored_df["is_anomaly"].astype(bool),
        "hybrid": scored_df["is_anomaly_hybrid"].astype(bool),
    }

    summary_rows = []
    confusion_rows = []

    for method_name, y_pred in methods.items():
        tp = int((y_true & y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        tn = int((~y_true & ~y_pred).sum())
        fn = int((y_true & ~y_pred).sum())

        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        f1 = _safe_ratio(2 * precision * recall, precision + recall)
        accuracy = _safe_ratio(tp + tn, len(scored_df))

        summary_rows.append(
            {
                "method": method_name,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "accuracy": round(accuracy, 4),
                "predicted_anomaly_rate": round(float(y_pred.mean()), 4),
                "ground_truth_anomaly_rate": round(float(y_true.mean()), 4),
            }
        )

        confusion_rows.extend(
            [
                {
                    "method": method_name,
                    "actual": "anomaly",
                    "predicted": "anomaly",
                    "count": tp,
                },
                {
                    "method": method_name,
                    "actual": "normal",
                    "predicted": "anomaly",
                    "count": fp,
                },
                {
                    "method": method_name,
                    "actual": "normal",
                    "predicted": "normal",
                    "count": tn,
                },
                {
                    "method": method_name,
                    "actual": "anomaly",
                    "predicted": "normal",
                    "count": fn,
                },
            ]
        )

    return pd.DataFrame(summary_rows), pd.DataFrame(confusion_rows)


def train_isolation_forest(
    cleaned_df: pd.DataFrame,
    contamination: float = 0.08,
    random_state: int = 42,
    window_days: int = 7,
    min_history: int = 5,
    employee_threshold: float = 2.0,
) -> Tuple[IsolationForest, pd.DataFrame]:
    """Train isolation forest and attach global + employee-level anomaly signals."""
    if cleaned_df.empty:
        raise ValueError("No valid records available for model training.")
    if len(cleaned_df) < 10:
        raise ValueError("At least 10 valid rows are required for stable training.")

    model_input = build_model_matrix(cleaned_df)
    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
    )
    predictions = model.fit_predict(model_input)
    raw_scores = model.score_samples(model_input)

    result_df = cleaned_df.copy()
    result_df["anomaly_score"] = -raw_scores
    result_df["is_anomaly"] = predictions == -1

    global_thresholds = _build_global_reason_thresholds(result_df)
    result_df["reason_global"] = result_df.apply(
        _reason_for_row_global,
        axis=1,
        args=(global_thresholds,),
    )

    result_df = _compute_employee_pattern_signals(
        result_df,
        window_days=window_days,
        min_history=min_history,
        threshold=employee_threshold,
    )

    result_df["is_anomaly_hybrid"] = (
        result_df["is_anomaly"] | result_df["is_anomaly_employee"]
    )

    global_severity = _normalize_series(result_df["anomaly_score"])
    employee_severity_norm = (result_df["severity_employee"] / 4.0).clip(0.0, 1.0)

    result_df["severity_hybrid"] = 0.0
    both_mask = result_df["is_anomaly"] & result_df["is_anomaly_employee"]
    global_only_mask = result_df["is_anomaly"] & ~result_df["is_anomaly_employee"]
    employee_only_mask = ~result_df["is_anomaly"] & result_df["is_anomaly_employee"]

    result_df.loc[both_mask, "severity_hybrid"] = (
        global_severity.loc[both_mask] + employee_severity_norm.loc[both_mask]
    ) / 2.0
    result_df.loc[global_only_mask, "severity_hybrid"] = global_severity.loc[
        global_only_mask
    ]
    result_df.loc[employee_only_mask, "severity_hybrid"] = employee_severity_norm.loc[
        employee_only_mask
    ]

    result_df["reason"] = result_df.apply(_combine_reason, axis=1)

    return model, result_df
