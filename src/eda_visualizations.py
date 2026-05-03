import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def run_eda(df):
    print("\n" + "="*60)
    print("DATA QUALITY REPORT")
    print("="*60)
    print(f"Records         : {len(df):,}")
    print(f"Features        : {len(df.columns)}")
    print(f"Missing Values  : {df.isnull().sum().sum()}")
    print(f"Duplicates      : {df.duplicated().sum()}")
    print("\nStatistical Summary:")
    print(df[['Age', 'Length_of_Stay_Days', 'Total_Cost_USD', 'Satisfaction_Score']].describe().round(2))

def plot_main_dashboard(df):
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle("Healthcare Analytics Dashboard — 2023", fontsize=18, fontweight='bold', y=0.98)

    ax1 = plt.subplot(4, 3, 1)
    for gender in ['Male', 'Female']:
        ax1.hist(df[df['Gender'] == gender]['Age'], bins=20, alpha=0.6, label=gender, edgecolor='black')
    ax1.set(xlabel='Age', ylabel='Count', title='Age Distribution by Gender')
    ax1.legend()

    ax2 = plt.subplot(4, 3, 2)
    age_counts = df['Age_Group'].value_counts().sort_index()
    bars = ax2.bar(age_counts.index, age_counts.values,
                   color=plt.cm.Set3(np.linspace(0, 1, len(age_counts))), edgecolor='black')
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=9)
    ax2.set(xlabel='Age Group', ylabel='Patients', title='Patients by Age Group')

    ax3 = plt.subplot(4, 3, 3)
    ins = df['Insurance_Type'].value_counts()
    ax3.pie(ins.values, labels=ins.index, autopct='%1.1f%%', startangle=90,
            colors=plt.cm.Pastel1(np.linspace(0, 1, len(ins))))
    ax3.set_title('Insurance Type Distribution')

    ax4 = plt.subplot(4, 3, 4)
    diag = df['Primary_Diagnosis'].value_counts().head(8)
    ax4.barh(diag.index[::-1], diag.values[::-1],
             color=plt.cm.Reds(np.linspace(0.3, 0.9, len(diag))))
    ax4.set(xlabel='Cases', title='Top Primary Diagnoses')

    ax5 = plt.subplot(4, 3, 5)
    dept = df['Department'].value_counts()
    ax5.bar(range(len(dept)), dept.values,
            color=plt.cm.viridis(np.linspace(0, 1, len(dept))), edgecolor='black')
    ax5.set_xticks(range(len(dept)))
    ax5.set_xticklabels(dept.index, rotation=45, ha='right', fontsize=9)
    ax5.set(ylabel='Patient Count', title='Department Utilization')

    ax6 = plt.subplot(4, 3, 6)
    month_order = ['January','February','March','April','May','June',
                   'July','August','September','October','November','December']
    adm_month = df.groupby(['Month', 'Admission_Type']).size().unstack(fill_value=0)
    adm_month = adm_month.reindex([m for m in month_order if m in adm_month.index])
    adm_month.plot(kind='bar', stacked=True, ax=ax6, colormap='tab10', width=0.8)
    ax6.set(xlabel='Month', ylabel='Admissions', title='Monthly Admissions by Type')
    ax6.legend(title='Type', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax6.tick_params(axis='x', rotation=45)

    ax7 = plt.subplot(4, 3, 7)
    dept_costs = df.groupby('Department')['Total_Cost_USD'].mean().sort_values(ascending=False)
    ax7.bar(dept_costs.index, dept_costs.values,
            color=plt.cm.plasma(np.linspace(0, 1, len(dept_costs))), edgecolor='black')
    ax7.set_xticklabels(dept_costs.index, rotation=45, ha='right', fontsize=9)
    ax7.set(ylabel='Avg Cost ($)', title='Average Cost by Department')
    ax7.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))

    ax8 = plt.subplot(4, 3, 8)
    sc = ax8.scatter(df['Length_of_Stay_Days'], df['Total_Cost_USD'],
                     c=df['Age'], cmap='coolwarm', alpha=0.5, s=20)
    plt.colorbar(sc, ax=ax8, label='Age')
    ax8.set(xlabel='Length of Stay (Days)', ylabel='Total Cost ($)', title='Cost vs LOS')

    ax9 = plt.subplot(4, 3, 9)
    ins_order = df['Insurance_Type'].unique()
    data_boxes = [df[df['Insurance_Type'] == i]['Total_Cost_USD'].values for i in ins_order]
    bp = ax9.boxplot(data_boxes, labels=ins_order, patch_artist=True)
    for patch, color in zip(bp['boxes'], plt.cm.Set2(np.linspace(0, 1, len(bp['boxes'])))):
        patch.set(facecolor=color, alpha=0.7)
    ax9.set(ylabel='Total Cost ($)', title='Cost by Insurance Type')
    ax9.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
    ax9.tick_params(axis='x', rotation=45)

    ax10 = plt.subplot(4, 3, 10)
    readmit = df.groupby('Primary_Diagnosis')['Readmitted_30_Days'].apply(
        lambda x: (x == 'Yes').sum() / len(x) * 100).sort_values(ascending=False)
    ax10.bar(readmit.index, readmit.values,
             color=plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(readmit))), edgecolor='black')
    ax10.axhline(readmit.mean(), color='red', linestyle='--', linewidth=2,
                 label=f'Avg: {readmit.mean():.1f}%')
    ax10.set_xticklabels(readmit.index, rotation=45, ha='right', fontsize=9)
    ax10.set(ylabel='Readmission Rate (%)', title='30-Day Readmission by Diagnosis')
    ax10.legend()

    ax11 = plt.subplot(4, 3, 11)
    ax11.hist(df['Satisfaction_Score'], bins=25, color='teal', alpha=0.7, edgecolor='black')
    ax11.axvline(df['Satisfaction_Score'].mean(), color='red', linestyle='--', linewidth=2,
                 label=f"Mean: {df['Satisfaction_Score'].mean():.1f}")
    ax11.set(xlabel='Score', ylabel='Frequency', title='Satisfaction Score Distribution')
    ax11.legend()

    ax12 = plt.subplot(4, 3, 12)
    dept_sat = df.groupby('Department')['Satisfaction_Score'].mean().sort_values(ascending=False)
    ax12.barh(dept_sat.index[::-1], dept_sat.values[::-1],
              color=plt.cm.Greens(np.linspace(0.3, 0.9, len(dept_sat))))
    ax12.set(xlabel='Avg Satisfaction Score', title='Satisfaction by Department', xlim=(0, 100))

    plt.tight_layout()
    plt.savefig("outputs/dashboard.png", dpi=150, bbox_inches='tight', facecolor='white')
    print("Dashboard saved to outputs/dashboard.png")
    plt.close()

def plot_time_trends(df):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Healthcare Trend Analysis — 2023", fontsize=16, fontweight='bold')

    monthly = df.groupby(df['Admission_Date'].dt.to_period('M')).size()
    ax1 = axes[0, 0]
    ax1.plot(range(len(monthly)), monthly.values, marker='o', linewidth=2, color='#2E86AB')
    ax1.fill_between(range(len(monthly)), monthly.values, alpha=0.3, color='#2E86AB')
    ax1.set_xticks(range(0, len(monthly), 2))
    ax1.set_xticklabels([str(monthly.index[i]) for i in range(0, len(monthly), 2)], rotation=45)
    ax1.set(title='Monthly Admission Volume', xlabel='Month', ylabel='Admissions')

    quarterly = df.groupby('Quarter')['Total_Cost_USD'].sum() / 1e6
    ax2 = axes[0, 1]
    bars = ax2.bar(quarterly.index, quarterly.values,
                   color=['#A23B72', '#F18F01', '#C73E1D', '#3B1F2B'], edgecolor='black')
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'${bar.get_height():.1f}M', ha='center', va='bottom', fontweight='bold')
    ax2.set(title='Quarterly Revenue', ylabel='Revenue ($ Millions)')

    ax3 = axes[1, 0]
    top3 = df['Primary_Diagnosis'].value_counts().head(3).index
    seasonal = df.groupby(['Month', 'Primary_Diagnosis']).size().unstack(fill_value=0)
    seasonal[top3].plot(kind='line', ax=ax3, marker='o', linewidth=2)
    ax3.set(title='Seasonal Trends: Top 3 Diagnoses', xlabel='Month', ylabel='Cases')
    ax3.legend(title='Diagnosis', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.tick_params(axis='x', rotation=45)

    ax4 = axes[1, 1]
    age_trends = df.groupby([df['Admission_Date'].dt.quarter, 'Age_Group']).size().unstack(fill_value=0)
    age_trends.plot(kind='bar', stacked=True, ax=ax4, colormap='tab20', width=0.7)
    ax4.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'], rotation=0)
    ax4.set(title='Age Distribution by Quarter', xlabel='Quarter', ylabel='Patients')
    ax4.legend(title='Age Group', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig("outputs/time_trends.png", dpi=150, bbox_inches='tight', facecolor='white')
    print("Time trends saved to outputs/time_trends.png")
    plt.close()
