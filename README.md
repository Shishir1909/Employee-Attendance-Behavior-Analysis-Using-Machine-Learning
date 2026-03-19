# Employee Attendance Behavior Analysis Using Machine Learning

This project detects abnormal employee attendance patterns using a hybrid approach:
- global Isolation Forest outlier detection
- employee-level 7-day rolling baseline deviation checks

Results and diagnostics are presented through a Streamlit web app.

## Problem Statement
Organizations collect attendance records (login/logout times), but raw logs do not directly explain suspicious work patterns.  
This solution processes attendance data, engineers features, trains an anomaly model, and highlights potential irregular behavior.

## Repository Contents
- `app.py`: Streamlit application.
- `src/pipeline.py`: schema validation, cleaning, and feature engineering.
- `src/modeling.py`: global model, employee-level pattern detection, hybrid decision, and ground-truth evaluation.
- `src/data_generator.py`: synthetic attendance dataset generator.

## Required Dataset Schema
CSV input must include:
- `employee_id`
- `login_time` (`HH:MM`)
- `logout_time` (`HH:MM`)
- `date` (`YYYY-MM-DD`)
- `department`
- `ground_truth_anomaly` (`0/1`)
- `ground_truth_type` (`normal`, `early_logout`, `late_login`, `short_duration`)

## Methodology
1. Validate required schema and normalize string columns.
2. Parse times into minutes (`login_minutes`, `logout_minutes`).
3. Drop invalid rows deterministically:
- missing required values
- invalid date/time formats
- logout earlier than or equal to login
- duplicate records
4. Engineer features:
- `total_work_minutes`
- `day_of_week`
- `login_deviation_from_9am`
- `is_weekend`
5. Train `IsolationForest` using numeric features + one-hot encoded department.
6. Build employee-specific baseline signals using a 7-day rolling history and robust scaling.
7. Final hybrid decision = global anomaly OR employee anomaly.
8. Output:
- `anomaly_score`
- `is_anomaly`
- `is_anomaly_employee`
- `is_anomaly_hybrid`
- `reason_global`, `reason_employee`, `reason`
- `severity_employee`, `severity_hybrid`
- optional evaluation metrics when `ground_truth_anomaly` exists

## ML Approach
- Global model: `IsolationForest` (unsupervised anomaly detection).
- Input features: engineered time and date indicators + encoded department.
- Employee-level detection:
- 7-day rolling baseline per employee for login/logout/duration
- robust z-like deviation score (IQR/std fallback)
- minimum history of 5 records before employee-level flagging
- Final anomaly uses a hybrid signal and prioritized reasoning.

## Quick Start
### 1) Create environment and install dependencies
```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Generate sample dataset
```powershell
python src/data_generator.py --output data/sample_attendance.csv
```

### 3) Run Streamlit app
```powershell
python -m streamlit run app.py
```

## Streamlit Sections Implemented
- Dataset Upload
- Data Preview
- Feature Engineering Output
- Model Training
- Results Visualization (global vs employee vs hybrid)
- Ground Truth Evaluation (when labels are available)
- Export to `output/`
