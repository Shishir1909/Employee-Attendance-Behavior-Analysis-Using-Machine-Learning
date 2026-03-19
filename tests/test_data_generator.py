from src.data_generator import generate_synthetic_attendance


def test_generator_emits_ground_truth_columns() -> None:
    df = generate_synthetic_attendance(num_employees=10, days=15, seed=7)

    assert "ground_truth_anomaly" in df.columns
    assert "ground_truth_type" in df.columns
    assert set(df["ground_truth_type"].unique()).issubset(
        {"normal", "early_logout", "late_login", "short_duration"}
    )


def test_ground_truth_binary_matches_type() -> None:
    df = generate_synthetic_attendance(num_employees=8, days=10, seed=13)

    derived = (df["ground_truth_type"] != "normal").astype(int)
    assert (derived == df["ground_truth_anomaly"]).all()

