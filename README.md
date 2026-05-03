# 🏥 Healthcare Analytics — End-to-End Data Analysis

A complete Python analytics pipeline across 5,000 patient records — covering data generation, exploratory analysis, a 12-panel visual dashboard, and a machine learning model for predicting 30-day readmission risk.

---

## 🎯 Business Questions Answered

| Question | Method |
|---|---|
| Which patients are most likely to be readmitted? | Random Forest Classifier |
| How do costs vary across departments and insurance types? | Group aggregation + box plots |
| Are there seasonal trends in hospital admissions? | Time series analysis |
| What drives patient satisfaction? | Correlation + visual analysis |
| Where should hospital resources be reallocated? | KPI dashboard |

---

## 📁 Project Structure

```
├── main.py                        ← Run this
├── src/
│   ├── data_generator.py          ← Generates 5,000 patient records
│   ├── eda_visualizations.py      ← 12-panel dashboard + time series
│   └── predictive_model.py        ← ML model + KPIs
├── data/                          ← Auto-created on first run
├── outputs/                       ← All charts saved here
└── requirements.txt
```

---

## ⚙️ Setup & Run

```bash
git clone https://github.com/Afreen00/Healthcare-analytics.git
cd Healthcare-analytics
pip install -r requirements.txt
python main.py
```

All output charts are saved automatically to `/outputs`.

---

## 📊 What Gets Generated

- **12-panel dashboard** — demographics, cost analysis, department utilization, readmission rates, satisfaction scores
- **Time series charts** — monthly admissions, quarterly revenue, seasonal disease patterns
- **ML model** — Random Forest classifier with ROC-AUC score, confusion matrix, feature importance
- **KPI summary** — printed to terminal on every run

---

## 📈 Key Results

| KPI | Value |
|---|---|
| Total Patients | 5,000 |
| Total Revenue | ~$45M |
| Avg Length of Stay | 4.2 days |
| 30-Day Readmission Rate | ~15.8% |
| Patient Satisfaction | 82.4/100 |
| Model ROC-AUC | ~0.74 |

---

## 🛠️ Tech Stack

Python | Pandas | NumPy | Matplotlib | Seaborn | Scikit-learn

---

## 👩‍💻 Author

**Afreen** — Operations Intelligence Analyst | AI Ops Specialist

[LinkedIn](https://www.linkedin.com/in/afreen-71ba6b226) · [GitHub](https://github.com/Afreen00)
