from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _minutes_to_hhmm(minutes: int) -> str:
    minutes = max(0, min(minutes, 23 * 60 + 59))
    hour = minutes // 60
    minute = minutes % 60
    return f"{hour:02d}:{minute:02d}"


def generate_synthetic_attendance(
    num_employees: int = 60,
    days: int = 45,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    departments = ["IT", "HR", "Finance", "Operations", "Sales", "Marketing"]
    employee_ids = [f"E{i:03d}" for i in range(1, num_employees + 1)]
    employee_dept = {
        employee_id: rng.choice(departments).item() for employee_id in employee_ids
    }

    start_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=days - 1)
    date_range = pd.date_range(start_date, periods=days, freq="D")

    rows: List[dict] = []
    for current_date in date_range:
        for employee_id in employee_ids:
            if rng.random() < 0.03:
                # Absent day.
                continue

            login_minutes = int(rng.normal(9 * 60, 18))
            logout_minutes = int(rng.normal(17 * 60 + 30, 22))

            if current_date.weekday() >= 5:
                login_minutes += int(rng.integers(-10, 25))
                logout_minutes -= int(rng.integers(15, 60))

            ground_truth_type = "normal"
            roll = rng.random()
            if roll < 0.05:
                # Early logout anomaly.
                logout_minutes -= int(rng.integers(90, 210))
                ground_truth_type = "early_logout"
            elif roll < 0.08:
                # Late login anomaly.
                login_minutes += int(rng.integers(60, 150))
                ground_truth_type = "late_login"
            elif roll < 0.10:
                # Short duration anomaly.
                logout_minutes = login_minutes + int(rng.integers(150, 300))
                ground_truth_type = "short_duration"

            login_minutes = int(np.clip(login_minutes, 7 * 60, 12 * 60))
            logout_minutes = int(np.clip(logout_minutes, login_minutes + 60, 21 * 60))

            rows.append(
                {
                    "employee_id": employee_id,
                    "login_time": _minutes_to_hhmm(login_minutes),
                    "logout_time": _minutes_to_hhmm(logout_minutes),
                    "date": current_date.strftime("%Y-%m-%d"),
                    "department": employee_dept[employee_id],
                    "ground_truth_anomaly": int(ground_truth_type != "normal"),
                    "ground_truth_type": ground_truth_type,
                }
            )

    df = pd.DataFrame(rows)

    # Add a few controlled dirty records so cleaning logic is visible in the app.
    if len(df) > 20:
        dirty_indices = rng.choice(df.index, size=5, replace=False)
        df.loc[dirty_indices[0], "login_time"] = "25:10"
        df.loc[dirty_indices[1], "logout_time"] = "not_a_time"
        df.loc[dirty_indices[2], "employee_id"] = ""
        df.loc[dirty_indices[3], "department"] = ""
        df.loc[dirty_indices[4], "logout_time"] = "05:00"

    # Add duplicate rows.
    if len(df) > 50:
        duplicate_sample = df.sample(n=8, random_state=seed)
        df = pd.concat([df, duplicate_sample], ignore_index=True)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic attendance dataset")
    parser.add_argument(
        "--output",
        default="data/sample_attendance.csv",
        help="Output CSV path",
    )
    parser.add_argument("--employees", type=int, default=60)
    parser.add_argument("--days", type=int, default=45)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = generate_synthetic_attendance(
        num_employees=args.employees,
        days=args.days,
        seed=args.seed,
    )
    dataset.to_csv(output_path, index=False)
    print(f"Synthetic dataset saved to: {output_path}")
    print(f"Rows: {len(dataset)}")


if __name__ == "__main__":
    main()
