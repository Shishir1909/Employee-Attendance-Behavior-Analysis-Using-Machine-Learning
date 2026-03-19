from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

REQUIRED_COLUMNS = [
    "employee_id",
    "login_time",
    "logout_time",
    "date",
    "department",
]


@dataclass
class CleaningReport:
    total_rows: int
    valid_rows: int
    dropped_rows: int
    duplicate_rows: int
    invalid_rows: int


def parse_time_to_minutes(value: object) -> Optional[int]:
    """Convert HH:MM text to minutes since midnight."""
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    parts = text.split(":")
    if len(parts) != 2:
        return None

    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except ValueError:
        return None

    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None

    return hour * 60 + minute


def validate_schema(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_text}")


def _is_missing_text(value: object) -> bool:
    return pd.isna(value) or str(value).strip() == ""


def _drop_reason(row: pd.Series) -> str:
    reasons = []

    if _is_missing_text(row["employee_id"]):
        reasons.append("missing_employee_id")
    if _is_missing_text(row["department"]):
        reasons.append("missing_department")
    if pd.isna(row["date_parsed"]):
        reasons.append("invalid_date")
    if pd.isna(row["login_minutes"]):
        reasons.append("invalid_login_time")
    if pd.isna(row["logout_minutes"]):
        reasons.append("invalid_logout_time")

    if (
        not pd.isna(row["login_minutes"])
        and not pd.isna(row["logout_minutes"])
        and row["logout_minutes"] <= row["login_minutes"]
    ):
        reasons.append("logout_before_or_equal_login")

    return ";".join(reasons)


def preprocess_attendance(
    raw_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, CleaningReport]:
    """Validate, clean, and engineer attendance features."""
    validate_schema(raw_df)
    working_df = raw_df.copy()
    optional_columns = [column for column in raw_df.columns if column not in REQUIRED_COLUMNS]

    for column in REQUIRED_COLUMNS:
        if working_df[column].dtype == "object":
            working_df[column] = working_df[column].astype(str).str.strip()
            working_df.loc[working_df[column] == "nan", column] = ""

    working_df["date_parsed"] = pd.to_datetime(working_df["date"], errors="coerce")
    working_df["login_minutes"] = working_df["login_time"].apply(parse_time_to_minutes)
    working_df["logout_minutes"] = working_df["logout_time"].apply(parse_time_to_minutes)
    working_df["drop_reason"] = working_df.apply(_drop_reason, axis=1)

    invalid_mask = working_df["drop_reason"] != ""
    invalid_rows = working_df.loc[
        invalid_mask, REQUIRED_COLUMNS + optional_columns + ["drop_reason"]
    ].copy()
    valid_df = working_df.loc[~invalid_mask].copy()

    duplicate_mask = valid_df.duplicated(subset=REQUIRED_COLUMNS, keep="first")
    duplicate_rows = valid_df.loc[duplicate_mask, REQUIRED_COLUMNS + optional_columns].copy()
    if not duplicate_rows.empty:
        duplicate_rows["drop_reason"] = "duplicate_record"
    valid_df = valid_df.loc[~duplicate_mask].copy()

    valid_df["date"] = valid_df["date_parsed"].dt.strftime("%Y-%m-%d")
    valid_df["day_of_week"] = valid_df["date_parsed"].dt.day_name()
    valid_df["is_weekend"] = (
        valid_df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)
    )
    valid_df["total_work_minutes"] = (
        valid_df["logout_minutes"] - valid_df["login_minutes"]
    )
    valid_df["login_deviation_from_9am"] = valid_df["login_minutes"] - 540

    valid_df = valid_df[
        REQUIRED_COLUMNS
        + optional_columns
        + [
            "login_minutes",
            "logout_minutes",
            "total_work_minutes",
            "day_of_week",
            "login_deviation_from_9am",
            "is_weekend",
        ]
    ].sort_values(by=["date", "employee_id", "login_time"])

    valid_df = valid_df.reset_index(drop=True)
    dropped_df = pd.concat([invalid_rows, duplicate_rows], ignore_index=True)

    report = CleaningReport(
        total_rows=len(raw_df),
        valid_rows=len(valid_df),
        dropped_rows=len(dropped_df),
        duplicate_rows=len(duplicate_rows),
        invalid_rows=len(invalid_rows),
    )

    return valid_df, dropped_df, report
