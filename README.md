# 🏥 Healthcare Analytics [End-to-End Data Analysis]

A complete Python analytics pipeline built across 5,000 synthetic patient records — covering data generation, exploratory data analysis, a 12-panel visual dashboard, and a machine learning model predicting 30-day hospital readmission risk.

---

## 👩‍💻 Author

*Afreen* — Operations Intelligence Analyst | AI Ops Specialist
[LinkedIn](https://www.linkedin.com/in/afreen-71ba6b226) · [GitHub](https://github.com/Afreen00)

---

## 🎯 Business Questions Answered

| Question | Method |
|---|---|
| Which patients are most likely to be readmitted? | Random Forest Classifier |
| How do costs vary across departments and insurance types? | Group aggregation + box plots |
| Are there seasonal trends in hospital admissions? | Time series analysis |
| What drives patient satisfaction scores? | Correlation + visual analysis |
| Where should hospital resources be reallocated? | KPI dashboard + recommendations |

---

## 📁 Project Structureba6b226) · [GitHub](https://github.com/Afreen00)

├── main.py                        ← Run this to execute full pipeline
├── src/
│   ├── data_generator.py          ← Generates 5,000 patient records
│   ├── eda_visualizations.py      ← 12-panel dashboard + time series
│   └── predictive_model.py        ← ML model + KPIs
├── data/                          ← Auto-created on first run
├── outputs/                       ← All charts saved here
└── requirements.txt

## ⚙️ Setup & Run

```bash
git clone https://github.com/Afreen00/Healthcare-analytics.git
cd Healthcare-analytics
pip install -r requirements.txt
python main.py

🧠 Key Code Highlights
1. Realistic Data Generation with correlated variables
base_los = {'Emergency': 4.5, 'Elective': 2.8, 'Urgent': 3.5}
age_factor = 1 + (age / 100) * 0.5
length_of_stay = base_los[admission_type] * age_factor

2.Feature Engineering
df['Age_Group'] = pd.cut(df['Age'], bins=[0,18,35,50,65,100],
                  labels=['0-18','19-35','36-50','51-65','65+'])
df['Quarter'] = df['Admission_Date'].dt.quarter.map({1:'Q1',2:'Q2',3:'Q3',4:'Q4'})

3.Random Forest Readmission Classifier
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)
--
📊 Dashboard Outputs (12 Panels)
Age distribution by gender
Patient segmentation by age group
Insurance type breakdown
Top 8 primary diagnoses
Department utilisation
Monthly admissions by type
Average cost per department
Length of Stay vs Cost scatter plot
Cost distribution by insurance type
30-day readmission rate by diagnosis
Patient satisfaction distribution
Satisfaction score by department
--
📈## 📈 Key Results

| KPI | Value |
|---|---|
| Total Patients | 5,000 |
| Total Revenue | ~$45M |
| Avg Length of Stay | 4.2 days |
| 30-Day Readmission Rate | ~15.8% |
| Patient Satisfaction Score | 82.4 / 100 |
| Emergency Admission Rate | ~40% |
| Model ROC-AUC Score | ~0.74 |


💡 Strategic Recommendations
Readmission Reduction — Heart Disease and Diabetes patients have the highest readmission rates. Implement post-discharge follow-up within 48 hours. Target: reduce from 15.8% to below 12%.
Cost Optimisation — Trauma and Emergency departments carry the highest average costs. Standardise clinical pathways for top diagnoses. Target: 8–10% reduction in cost-per-case.
Capacity Planning — Q4 sees peak admissions. Pre-staff Cardiology and Orthopedics ahead of the quarter to reduce wait times.
Quality Improvement — Patient satisfaction is inversely correlated with length of stay. Improve discharge planning processes. Target: increase satisfaction from 82.4 to above 90.
Population Health — Patients aged 65+ consume the most resources. Launch preventive care programmes to reduce emergency conversions.
--
🛠️ Tech Stack
Python | Pandas | NumPy | Matplotlib | Seaborn | Scikit-learn
