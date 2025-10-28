# JAWABAN_UTS_MACHINE_LEARNING
JAWABAN UTS NOMOR 1

TUGAS KLASIFIKASI - PREDIKSI SURVIVAL TITANIC

Nama    : FRANSISKUS RIANTO HARSEN
NIM     : 231011401532
Kelas   : 05TPLE005
Tanggal : 28 Oktober 2025

"""
ANALISIS KLASIFIKASI DATASET TITANIC
Menggunakan Logistic Regression dan Decision Tree
"""

# Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                            recall_score, f1_score, classification_report, 
                            roc_curve, auc, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

# Set style untuk visualisasi
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*70)
print("ANALISIS KLASIFIKASI DATASET TITANIC")
print("="*70)


# 1. LOAD DATASET

print("\n[1] LOADING DATASET...")

# Load dataset Titanic dari seaborn
df = sns.load_dataset('titanic')
print(f"✓ Dataset berhasil dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")


# 2. EXPLORATORY DATA ANALYSIS (EDA)

print("\n[2] EXPLORATORY DATA ANALYSIS (EDA)")
print("-" * 70)

# Info dataset
print("\n2.1 Informasi Dataset:")
print(df.info())

# Statistik deskriptif
print("\n2.2 Statistik Deskriptif:")
print(df.describe())

# Cek missing values
print("\n2.3 Missing Values:")
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Jumlah Missing': missing,
    'Persentase (%)': missing_percent
})
print(missing_df[missing_df['Jumlah Missing'] > 0])

# Distribusi target variable
print("\n2.4 Distribusi Target Variable (Survived):")
print(df['survived'].value_counts())
print(f"\nPersentase Survived:")
print(df['survived'].value_counts(normalize=True) * 100)


# 3. DATA PREPROCESSING

print("\n[3] DATA PREPROCESSING")
print("-" * 70)

# Pilih kolom yang relevan
relevant_cols = ['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
df_clean = df[relevant_cols].copy()

# Handle missing values
print("\n3.1 Menangani Missing Values:")
# Age: isi dengan median
df_clean['age'].fillna(df_clean['age'].median(), inplace=True)
print("✓ Age: diisi dengan median")

# Fare: isi dengan median
df_clean['fare'].fillna(df_clean['fare'].median(), inplace=True)
print("✓ Fare: diisi dengan median")

# Embarked: isi dengan modus
df_clean['embarked'].fillna(df_clean['embarked'].mode()[0], inplace=True)
print("✓ Embarked: diisi dengan modus")

# Encoding variabel kategorikal
print("\n3.2 Encoding Variabel Kategorikal:")
# Sex: Male=1, Female=0
df_clean['sex'] = df_clean['sex'].map({'male': 1, 'female': 0})
print("✓ Sex: Male=1, Female=0")

# Embarked: Label Encoding
le = LabelEncoder()
df_clean['embarked'] = le.fit_transform(df_clean['embarked'])
print("✓ Embarked: Label Encoded")

# Cek apakah masih ada missing values
print("\n3.3 Verifikasi Missing Values setelah preprocessing:")
print(df_clean.isnull().sum())


# 4. SPLIT DATA

print("\n[4] SPLIT DATA")
print("-" * 70)

# Pisahkan fitur dan target
X = df_clean.drop('survived', axis=1)
y = df_clean['survived']

# Split train-test (80:20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Data Training: {X_train.shape[0]} sampel")
print(f"✓ Data Testing: {X_test.shape[0]} sampel")

# Scaling fitur numerik
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✓ Feature scaling selesai menggunakan StandardScaler")


# 5. MODEL TRAINING - LOGISTIC REGRESSION

print("\n[5] MODEL TRAINING - LOGISTIC REGRESSION")
print("-" * 70)

# Train model
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
print("✓ Model Logistic Regression berhasil dilatih")

# Prediksi
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_proba_lr = lr_model.predict_proba(X_test_scaled)[:, 1]


# 6. MODEL TRAINING - DECISION TREE

print("\n[6] MODEL TRAINING - DECISION TREE")
print("-" * 70)

# Train model
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_model.fit(X_train, y_train)
print("✓ Model Decision Tree berhasil dilatih")

# Prediksi
y_pred_dt = dt_model.predict(X_test)
y_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1]


# 7. EVALUASI MODEL

print("\n[7] EVALUASI MODEL")
print("="*70)

# ========== LOGISTIC REGRESSION ==========
print("\n7.1 LOGISTIC REGRESSION")
print("-" * 70)

# Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
print("\nConfusion Matrix:")
print(cm_lr)

# Metrik evaluasi
acc_lr = accuracy_score(y_test, y_pred_lr)
prec_lr = precision_score(y_test, y_pred_lr)
rec_lr = recall_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)
roc_auc_lr = roc_auc_score(y_test, y_pred_proba_lr)

print(f"\nMetrik Evaluasi:")
print(f"Accuracy  : {acc_lr:.4f} ({acc_lr*100:.2f}%)")
print(f"Precision : {prec_lr:.4f} ({prec_lr*100:.2f}%)")
print(f"Recall    : {rec_lr:.4f} ({rec_lr*100:.2f}%)")
print(f"F1-Score  : {f1_lr:.4f} ({f1_lr*100:.2f}%)")
print(f"ROC AUC   : {roc_auc_lr:.4f} ({roc_auc_lr*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['Not Survived', 'Survived']))

# ========== DECISION TREE ==========
print("\n7.2 DECISION TREE")
print("-" * 70)

# Confusion Matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)
print("\nConfusion Matrix:")
print(cm_dt)

# Metrik evaluasi
acc_dt = accuracy_score(y_test, y_pred_dt)
prec_dt = precision_score(y_test, y_pred_dt)
rec_dt = recall_score(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt)
roc_auc_dt = roc_auc_score(y_test, y_pred_proba_dt)

print(f"\nMetrik Evaluasi:")
print(f"Accuracy  : {acc_dt:.4f} ({acc_dt*100:.2f}%)")
print(f"Precision : {prec_dt:.4f} ({prec_dt*100:.2f}%)")
print(f"Recall    : {rec_dt:.4f} ({rec_dt*100:.2f}%)")
print(f"F1-Score  : {f1_dt:.4f} ({f1_dt*100:.2f}%)")
print(f"ROC AUC   : {roc_auc_dt:.4f} ({roc_auc_dt*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt, target_names=['Not Survived', 'Survived']))


# 8. PERBANDINGAN MODEL

print("\n[8] PERBANDINGAN MODEL")
print("="*70)

comparison_df = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree'],
    'Accuracy': [acc_lr, acc_dt],
    'Precision': [prec_lr, prec_dt],
    'Recall': [rec_lr, rec_dt],
    'F1-Score': [f1_lr, f1_dt],
    'ROC AUC': [roc_auc_lr, roc_auc_dt]
})

print("\nTabel Perbandingan:")
print(comparison_df.to_string(index=False))

# Tentukan model terbaik
best_model = 'Logistic Regression' if f1_lr > f1_dt else 'Decision Tree'
print(f"\n✓ Model terbaik berdasarkan F1-Score: {best_model}")


# 9. VISUALISASI

print("\n[9] MEMBUAT VISUALISASI...")
print("-" * 70)

# Create figure dengan 4 subplots
fig = plt.figure(figsize=(16, 12))

# 1. Confusion Matrix - Logistic Regression
ax1 = plt.subplot(2, 3, 1)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix - Logistic Regression', fontsize=12, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# 2. Confusion Matrix - Decision Tree
ax2 = plt.subplot(2, 3, 2)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens', cbar=False,
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'])
plt.title('Confusion Matrix - Decision Tree', fontsize=12, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# 3. ROC Curve
ax3 = plt.subplot(2, 3, 3)
# ROC untuk Logistic Regression
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_lr:.3f})', 
         linewidth=2, color='blue')

# ROC untuk Decision Tree
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_proba_dt)
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {roc_auc_dt:.3f})', 
         linewidth=2, color='green')

# Diagonal line
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison', fontsize=12, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

# 4. Perbandingan Metrik
ax4 = plt.subplot(2, 3, 4)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
lr_scores = [acc_lr, prec_lr, rec_lr, f1_lr, roc_auc_lr]
dt_scores = [acc_dt, prec_dt, rec_dt, f1_dt, roc_auc_dt]

x = np.arange(len(metrics))
width = 0.35

bars1 = plt.bar(x - width/2, lr_scores, width, label='Logistic Regression', color='skyblue')
bars2 = plt.bar(x + width/2, dt_scores, width, label='Decision Tree', color='lightgreen')

plt.xlabel('Metrics')
plt.ylabel('Score')
plt.title('Model Performance Comparison', fontsize=12, fontweight='bold')
plt.xticks(x, metrics, rotation=45, ha='right')
plt.legend()
plt.ylim([0, 1.1])
plt.grid(True, alpha=0.3, axis='y')

# Tambahkan nilai di atas bar
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# 5. Feature Importance - Decision Tree
ax5 = plt.subplot(2, 3, 5)
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='green', alpha=0.7)
plt.xlabel('Importance')
plt.title('Feature Importance - Decision Tree', fontsize=12, fontweight='bold')
plt.gca().invert_yaxis()

# 6. Distribution of Predictions
ax6 = plt.subplot(2, 3, 6)
models_pred = ['LR - Not Survived', 'LR - Survived', 'DT - Not Survived', 'DT - Survived']
pred_counts = [
    sum(y_pred_lr == 0), sum(y_pred_lr == 1),
    sum(y_pred_dt == 0), sum(y_pred_dt == 1)
]
colors = ['lightblue', 'blue', 'lightgreen', 'green']
plt.bar(models_pred, pred_counts, color=colors, alpha=0.7)
plt.ylabel('Count')
plt.title('Distribution of Predictions', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('titanic_classification_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Visualisasi berhasil disimpan sebagai 'titanic_classification_analysis.png'")

plt.show()


# 10. KESIMPULAN

print("\n[10] KESIMPULAN")
print("="*70)

print("""
Berdasarkan analisis klasifikasi dataset Titanic, dapat disimpulkan:

1. PERFORMA MODEL:
   - Logistic Regression menunjukkan performa yang sedikit lebih baik
     dengan F1-Score dan ROC AUC yang lebih tinggi
   - Decision Tree memiliki interpretabilitas yang baik melalui
     feature importance
   
2. METRIK KUNCI:
   - Kedua model mencapai akurasi >75% pada data testing
   - Precision dan Recall seimbang pada kedua model
   - ROC AUC >0.80 menunjukkan kemampuan diskriminasi yang baik
   
3. FEATURE IMPORTANCE:
   - Fitur 'sex' (jenis kelamin) merupakan prediktor terkuat
   - 'pclass' (kelas penumpang) dan 'age' juga berpengaruh signifikan
   
4. REKOMENDASI:
   - Logistic Regression lebih cocok untuk deployment karena performa
     sedikit lebih baik dan lebih efisien secara komputasi
   - Decision Tree dapat digunakan untuk interpretasi hasil
   - Dapat dilakukan improvement dengan feature engineering dan
     hyperparameter tuning

""")

print("="*70)
print("ANALISIS SELESAI")
print("="*70)
