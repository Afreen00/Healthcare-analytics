import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_healthcare_data(n_patients=5000, seed=42):
    np.random.seed(seed)

    patient_ids = [f'P{str(i).zfill(5)}' for i in range(1, n_patients + 1)]
    ages = np.clip(np.random.normal(52, 18, n_patients).astype(int), 0, 95)
    genders = np.random.choice(['Male', 'Female'], n_patients, p=[0.48, 0.52])
    insurance_types = np.random.choice(
        ['Private', 'Medicare', 'Medicaid', 'Self-Pay', 'Other'],
        n_patients, p=[0.35, 0.30, 0.15, 0.12, 0.08]
    )
    admission_types = np.random.choice(
        ['Emergency', 'Elective', 'Urgent', 'Newborn', 'Trauma'],
        n_patients, p=[0.40, 0.25, 0.20, 0.10, 0.05]
    )
    departments = np.random.choice(
        ['Cardiology', 'Orthopedics', 'General Medicine', 'Oncology',
         'Neurology', 'Pediatrics', 'Emergency', 'Surgery'],
        n_patients, p=[0.18, 0.15, 0.20, 0.12, 0.10, 0.10, 0.10, 0.05]
    )
    diagnoses = np.random.choice(
        ['Hypertension', 'Diabetes', 'Heart Disease', 'Pneumonia',
         'Fracture', 'Cancer', 'Stroke', 'Asthma', 'Infection', 'Other'],
        n_patients, p=[0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05, 0.03, 0.02]
    )

    base_los = {'Emergency': 4.5, 'Elective': 2.8, 'Urgent': 3.5, 'Newborn': 2.5, 'Trauma': 6.2}
    los = []
    for i, adm_type in enumerate(admission_types):
        base = base_los[adm_type]
        age_factor = 1 + (ages[i] / 100) * 0.5
        stay = max(1, base * age_factor + np.random.normal(0, 1.5))
        los.append(round(stay, 1))

    daily_cost = np.random.choice([1200, 1800, 2500, 3200, 4500], n_patients)
    total_costs = [round(los[i] * daily_cost[i] * (1 + np.random.normal(0, 0.1)), 2) for i in range(n_patients)]

    readmission_prob = {
        'Heart Disease': 0.25, 'Diabetes': 0.22, 'Hypertension': 0.18,
        'Cancer': 0.20, 'Stroke': 0.24, 'Pneumonia': 0.15,
        'Fracture': 0.08, 'Asthma': 0.12, 'Infection': 0.10, 'Other': 0.10
    }
    readmitted = [
        np.random.choice(['Yes', 'No'], p=[readmission_prob[d], 1 - readmission_prob[d]])
        for d in diagnoses
    ]

    satisfaction = [round(min(100, max(1, 85 - (s * 2) + np.random.normal(0, 8))), 1) for s in los]
    start_date = datetime(2023, 1, 1)
    admission_dates = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_patients)]

    df = pd.DataFrame({
        'Patient_ID': patient_ids,
        'Age': ages,
        'Gender': genders,
        'Insurance_Type': insurance_types,
        'Admission_Type': admission_types,
        'Department': departments,
        'Primary_Diagnosis': diagnoses,
        'Length_of_Stay_Days': los,
        'Total_Cost_USD': total_costs,
        'Readmitted_30_Days': readmitted,
        'Satisfaction_Score': satisfaction,
        'Admission_Date': admission_dates
    })

    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 35, 50, 65, 100],
                              labels=['0-18', '19-35', '36-50', '51-65', '65+'])
    df['Month'] = df['Admission_Date'].dt.month_name()
    df['Quarter'] = df['Admission_Date'].dt.quarter.map({1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'})

    return df
