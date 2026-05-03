import os
import pandas as pd

os.makedirs("outputs", exist_ok=True)
os.makedirs("data", exist_ok=True)

print("="*60)
print("  HEALTHCARE ANALYTICS PIPELINE")
print("="*60)

print("\n[1/3] Generating dataset...")
from src.data_generator import generate_healthcare_data
df = generate_healthcare_data(n_patients=5000, seed=42)
df.to_csv("data/healthcare_data.csv", index=False)
print(f"  {len(df):,} records saved to data/healthcare_data.csv")

print("\n[2/3] Running EDA and generating charts...")
df = pd.read_csv("data/healthcare_data.csv", parse_dates=['Admission_Date'])
from src.eda_visualizations import run_eda, plot_main_dashboard, plot_time_trends
run_eda(df)
plot_main_dashboard(df)
plot_time_trends(df)

print("\n[3/3] Training readmission risk model...")
from src.predictive_model import train_model, print_kpis
train_model(df)
print_kpis(df)

print("\n" + "="*60)
print("  DONE. Check the /outputs folder for all charts.")
print("="*60)
