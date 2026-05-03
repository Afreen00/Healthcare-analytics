import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings('ignore')

def train_model(df):
    model_df = df.copy()
    categorical_cols = ['Gender', 'Insurance_Type', 'Admission_Type', 'Department', 'Primary_Diagnosis']
    for col in categorical_cols:
        le = LabelEncoder()
        model_df[f'{col}_enc'] = le.fit_transform(model_df[col])

    enc_cols = [f'{col}_enc' for col in categorical_cols]
    numeric_features = ['Age', 'Length_of_Stay_Days', 'Total_Cost_USD', 'Satisfaction_Score']
    feature_cols = numeric_features + enc_cols

    X = model_df[feature_cols]
    y = (model_df['Readmitted_30_Days'] == 'Yes').astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n" + "="*60)
    print("MODEL PERFORMANCE — READMISSION RISK CLASSIFIER")
    print("="*60)
    print(f"ROC-AUC Score : {roc_auc_score(y_test, y_proba):.3f}")
    print(classification_report(y_test, y_pred, target_names=['No Readmission', 'Readmission']))

    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 7))
    top_fi = feature_importance.head(10)
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(top_fi)))[::-1]
    bars = ax.barh(top_fi['Feature'][::-1], top_fi['Importance'][::-1], color=colors, edgecolor='black')
    for bar, val in zip(bars, top_fi['Importance'][::-1]):
        ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
    ax.set(xlabel='Importance Score', title='Top 10 Factors Predicting Hospital Readmission')
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150, bbox_inches='tight', facecolor='white')
    print("Feature importance saved to outputs/feature_importance.png")
    plt.close()

    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred,
                                             display_labels=['No Readmission', 'Readmission'],
                                             cmap='Blues', ax=ax2)
    ax2.set_title('Confusion Matrix', fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150, bbox_inches='tight', facecolor='white')
    print("Confusion matrix saved to outputs/confusion_matrix.png")
    plt.close()

    return model, feature_importance

def print_kpis(df):
    readmission_rate = (df['Readmitted_30_Days'] == 'Yes').mean() * 100
    avg_satisfaction = df['Satisfaction_Score'].mean()

    print("\n" + "="*60)
    print("KEY PERFORMANCE INDICATORS")
    print("="*60)
    print(f"  Total Patients          : {len(df):,}")
    print(f"  Total Revenue           : ${df['Total_Cost_USD'].sum():,.0f}")
    print(f"  Avg Length of Stay      : {df['Length_of_Stay_Days'].mean():.1f} days")
    print(f"  30-Day Readmission Rate : {readmission_rate:.1f}%")
    print(f"  Patient Satisfaction    : {avg_satisfaction:.1f}/100")
    print(f"  Emergency Admission %   : {(df['Admission_Type']=='Emergency').mean()*100:.1f}%")
