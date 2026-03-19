# Gamma Presentation Template (Final_Round Assignment)

Use this directly in Gamma. It follows the 7-slide outline required in `Final_Round.pdf`.

---

## Copy-Paste Prompt For Gamma

Create a clean, professional 7-slide presentation titled **"Employee Attendance Anomaly Detection using Hybrid Intelligence"**.

Audience: technical interviewer + hiring panel.  
Tone: clear, practical, data-driven.  
Visual style: modern, minimal, blue/teal color palette, readable charts/tables.  
Use one key chart/visual per slide wherever possible.

Use exactly these 7 slides:

1. Problem Understanding  
2. Dataset and Preprocessing Approach  
3. Feature Engineering Strategy  
4. Machine Learning Model Used  
5. Application Architecture  
6. Demonstration of Streamlit App  
7. Key Insights from Results

Use the following project facts and metrics:

- Dataset type: synthetic employee attendance data
- Raw rows: 2629
- Valid rows after preprocessing: 2616
- Dropped rows: 13 (Invalid: 5, Duplicates: 8)
- Contamination rate: 0.08
- Global anomalies: 210 (8.03%)
- Employee-pattern anomalies: 784 (29.97%)
- Hybrid anomalies: 810 (30.96%)

Evaluation summary:
- Global: precision 0.8524, recall 0.6938, F1 0.7650, accuracy 0.9580
- Hybrid: precision 0.3000, recall 0.9419, F1 0.4551, accuracy 0.7775

Confusion matrix counts:
- Global: TP 179, FP 31, TN 2327, FN 79
- Hybrid: TP 243, FP 567, TN 1791, FN 15

Top employees by anomalies:
- Global top: E032(9), E057(8), E009(8), E016(7), E035(7)
- Employee-pattern top: E016(18), E057(17), E025(17), E034(16), E059(16)
- Hybrid top: E016(19), E057(18), E025(17), E018(16), E008(16)

Add short speaker notes per slide.

---

## Slide-by-Slide Content Draft

### Slide 1 — Problem Understanding
**Title:** Why Attendance Intelligence Matters  
**Bullets:**
- Raw login/logout logs exist, but behavior-level insights are missing.
- HR needs automatic detection of unusual work patterns.
- Target patterns: late login, early logout, short work duration.
- Objective: detect anomalies and explain them with actionable reasons.
**Visual suggestion:** Simple workflow icon (Raw Logs -> Detection -> Insights).

### Slide 2 — Dataset and Preprocessing Approach
**Title:** Dataset and Data Cleaning Pipeline  
**Bullets:**
- Source: synthetic employee attendance records.
- Rows: 2629 raw -> 2616 valid after cleaning.
- Removed rows: 13 (invalid format/missing values/duplicates).
- Time conversion: `HH:MM` -> minutes since midnight.
- Data quality checks: invalid times, logout <= login, missing fields, duplicates.
**Visual suggestion:** Before/after data quality table + row-count funnel.

### Slide 3 — Feature Engineering Strategy
**Title:** Features Built for Pattern Detection  
**Bullets:**
- Core features: `login_minutes`, `logout_minutes`, `total_work_minutes`.
- Context features: `day_of_week`, `is_weekend`, `login_deviation_from_9am`.
- Employee-level rolling features (7-day window):
- `login_dev_emp`, `logout_dev_emp`, `duration_dev_emp`
- baseline confidence using minimum 5 prior records.
**Visual suggestion:** Feature table with example row transformations.

### Slide 4 — Machine Learning Model Used
**Title:** Hybrid Anomaly Detection Logic  
**Bullets:**
- Global model: Isolation Forest (contamination = 0.08).
- Employee model: 7-day personal baseline deviation scoring.
- Final decision: `is_anomaly_hybrid = is_anomaly OR is_anomaly_employee`.
- Output explainability:
- `reason_global`, `reason_employee`, final `reason`
- `severity_employee`, `severity_hybrid`.
**Visual suggestion:** Side-by-side model diagram + OR-combination logic box.

### Slide 5 — Application Architecture
**Title:** End-to-End Streamlit Architecture  
**Bullets:**
- Input: CSV upload or sample synthetic dataset.
- Pipeline: preprocess -> feature engineering -> model inference.
- Outputs:
- global/employee/hybrid anomaly counts
- trend charts and employee ranking charts
- row-level anomaly table with reasons/severity
- ground-truth evaluation and CSV export to `output/`.
**Visual suggestion:** System architecture block diagram.

### Slide 6 — Demonstration of Streamlit App
**Title:** Live App Walkthrough  
**Bullets:**
- Step 1: Upload dataset and inspect quality summary.
- Step 2: Train hybrid model and compare global vs employee vs hybrid counts.
- Step 3: Review charts:
- work duration distribution
- anomaly trends over time
- anomaly counts by employee
- Step 4: Export scored results and evaluation tables.
**Visual suggestion:** 3 screenshot strip (Upload, Results, Evaluation/Export).

### Slide 7 — Key Insights from Results
**Title:** Results and Tradeoff Insights  
**Bullets:**
- Global model is precise: P=0.8524, R=0.6938, F1=0.7650.
- Hybrid model is high-recall: P=0.3000, R=0.9419, F1=0.4551.
- Tradeoff: Hybrid catches more true anomalies (lower FN) but raises more false alarms (higher FP).
- Operational guidance:
- Use global for cleaner alerts.
- Use hybrid for investigation-focused monitoring.
- Future work: threshold tuning, per-department calibration, alert ranking.
**Visual suggestion:** Precision-recall tradeoff chart + confusion matrix snippet.

---

## Optional Closing Line

"This solution combines global outlier intelligence with individual behavior baselines, producing explainable anomaly flags suitable for both operational monitoring and deeper HR investigations."

