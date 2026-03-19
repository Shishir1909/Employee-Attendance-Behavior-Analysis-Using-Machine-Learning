import pandas as pd

from src.modeling import evaluate_with_ground_truth, train_isolation_forest
from src.pipeline import preprocess_attendance


def _build_employee_sequence() -> pd.DataFrame:
    rows = []
    for day in range(1, 13):
        rows.append(
            {
                "employee_id": "E001",
                "login_time": "09:00" if day <= 10 else "11:30",
                "logout_time": "17:30" if day <= 10 else "17:00",
                "date": f"2026-03-{day:02d}",
                "department": "IT",
                "ground_truth_anomaly": 0 if day <= 10 else 1,
                "ground_truth_type": "normal" if day <= 10 else "late_login",
            }
        )
    return pd.DataFrame(rows)


def test_train_isolation_forest_outputs_expected_columns() -> None:
    raw_df = _build_employee_sequence()
    cleaned_df, _, _ = preprocess_attendance(raw_df)

    _, scored_df = train_isolation_forest(cleaned_df, contamination=0.1, random_state=42)

    required_columns = {
        "anomaly_score",
        "is_anomaly",
        "reason_global",
        "login_dev_emp",
        "logout_dev_emp",
        "duration_dev_emp",
        "employee_history_count",
        "is_anomaly_employee",
        "reason_employee",
        "severity_employee",
        "is_anomaly_hybrid",
        "severity_hybrid",
        "reason",
    }
    assert required_columns.issubset(set(scored_df.columns))
    assert scored_df["is_anomaly"].dtype == bool
    assert scored_df["is_anomaly_employee"].dtype == bool
    assert scored_df["is_anomaly_hybrid"].dtype == bool


def test_employee_pattern_signal_triggers_for_late_login_sequence() -> None:
    raw_df = _build_employee_sequence()
    cleaned_df, _, _ = preprocess_attendance(raw_df)

    _, scored_df = train_isolation_forest(cleaned_df, contamination=0.05, random_state=42)
    flagged = scored_df.loc[scored_df["date"] == "2026-03-11"].iloc[0]

    assert flagged["employee_history_count"] >= 5
    assert bool(flagged["is_anomaly_employee"]) is True
    assert flagged["reason_employee"] in {
        "late_login_pattern",
        "short_duration_pattern",
        "combined_deviation_emp",
    }
    assert bool(flagged["is_anomaly_hybrid"]) is True


def test_ground_truth_evaluation_returns_global_and_hybrid_rows() -> None:
    raw_df = _build_employee_sequence()
    cleaned_df, _, _ = preprocess_attendance(raw_df)
    _, scored_df = train_isolation_forest(cleaned_df, contamination=0.1, random_state=42)

    summary_df, confusion_df = evaluate_with_ground_truth(scored_df)

    assert set(summary_df["method"].tolist()) == {"global", "hybrid"}
    assert set(confusion_df["method"].tolist()) == {"global", "hybrid"}
    assert {"precision", "recall", "f1", "accuracy"}.issubset(summary_df.columns)
