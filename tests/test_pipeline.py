import pandas as pd

from src.pipeline import parse_time_to_minutes, preprocess_attendance


def test_parse_time_to_minutes() -> None:
    assert parse_time_to_minutes("09:00") == 540
    assert parse_time_to_minutes("17:30") == 1050
    assert parse_time_to_minutes("25:00") is None
    assert parse_time_to_minutes("") is None


def test_preprocess_handles_invalid_and_duplicates_deterministically() -> None:
    raw_df = pd.DataFrame(
        [
            {
                "employee_id": "E001",
                "login_time": "09:00",
                "logout_time": "17:30",
                "date": "2026-03-10",
                "department": "IT",
            },
            {
                "employee_id": "E001",
                "login_time": "09:00",
                "logout_time": "17:30",
                "date": "2026-03-10",
                "department": "IT",
            },
            {
                "employee_id": "E002",
                "login_time": "not_a_time",
                "logout_time": "17:30",
                "date": "2026-03-10",
                "department": "HR",
            },
            {
                "employee_id": "E003",
                "login_time": "10:00",
                "logout_time": "09:00",
                "date": "2026-03-10",
                "department": "Finance",
            },
        ]
    )

    cleaned_df, dropped_df, report = preprocess_attendance(raw_df)

    assert report.total_rows == 4
    assert report.valid_rows == 1
    assert report.invalid_rows == 2
    assert report.duplicate_rows == 1
    assert report.dropped_rows == 3
    assert cleaned_df.iloc[0]["total_work_minutes"] == 510
    assert cleaned_df.iloc[0]["login_minutes"] == 540
    assert cleaned_df.iloc[0]["logout_minutes"] == 1050
    assert set(dropped_df["drop_reason"].tolist()) == {
        "duplicate_record",
        "invalid_login_time",
        "logout_before_or_equal_login",
    }


def test_preprocess_preserves_optional_ground_truth_columns() -> None:
    raw_df = pd.DataFrame(
        [
            {
                "employee_id": "E001",
                "login_time": "09:00",
                "logout_time": "17:30",
                "date": "2026-03-10",
                "department": "IT",
                "ground_truth_anomaly": 0,
                "ground_truth_type": "normal",
            }
        ]
    )

    cleaned_df, _, report = preprocess_attendance(raw_df)

    assert report.valid_rows == 1
    assert "ground_truth_anomaly" in cleaned_df.columns
    assert "ground_truth_type" in cleaned_df.columns
    assert int(cleaned_df.iloc[0]["ground_truth_anomaly"]) == 0
    assert cleaned_df.iloc[0]["ground_truth_type"] == "normal"
