# Attendance Anomaly Detection Presentation (7 Slides)

## Slide 1: Problem Understanding
- Attendance logs capture raw events but not behavior intelligence.
- Goal: detect suspicious attendance patterns automatically.
- Examples: short work duration, early logout, irregular day-to-day behavior.

## Slide 2: Dataset and Preprocessing
- Input schema: employee_id, login_time, logout_time, date, department.
- Cleaning rules:
- remove missing/invalid values
- reject logout <= login
- remove duplicates
- convert times to minutes since midnight.

## Slide 3: Feature Engineering Strategy
- Engineered features:
- login_minutes
- logout_minutes
- total_work_minutes
- day_of_week
- login_deviation_from_9am
- is_weekend
- Department encoded with one-hot encoding.

## Slide 4: Model Approach
- Model: Isolation Forest (unsupervised anomaly detection).
- Why: no labeled anomaly ground truth required.
- Output fields:
- anomaly_score
- is_anomaly
- reason (short_duration, early_logout, combined_deviation).

## Slide 5: Application Architecture (Streamlit)
- Dataset Upload section with schema validation.
- Data Preview for raw and cleaned datasets.
- Feature Engineering Output section for transformed columns.
- Model Training button with contamination control.
- Results Visualization dashboards.

## Slide 6: Demo and Results
- Show sample upload and training.
- Display anomaly summary metrics.
- Show charts:
- duration distribution
- anomaly count by employee
- anomaly trend over time
- Show anomaly table with reasons.

## Slide 7: Key Insights and Next Steps
- Patterns identified quickly without labels.
- Useful for HR/operations monitoring and exception triage.
- Future improvements:
- model comparison
- explainability layer
- role/shift-aware thresholds
- alert integration and scheduled reporting.

