import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. GENERATE SYNTHETIC DATASET
# ─────────────────────────────────────────────
def generate_data(n=1000, seed=42):
    np.random.seed(seed)
    data = {
        'Gender':             np.random.choice(['Male', 'Female'], n, p=[0.65, 0.35]),
        'Married':            np.random.choice(['Yes', 'No'], n, p=[0.65, 0.35]),
        'Dependents':         np.random.choice(['0','1','2','3+'], n, p=[0.57,0.17,0.16,0.10]),
        'Education':          np.random.choice(['Graduate','Not Graduate'], n, p=[0.78, 0.22]),
        'Self_Employed':      np.random.choice(['Yes', 'No'], n, p=[0.14, 0.86]),
        'ApplicantIncome':    np.random.lognormal(8.5, 0.6, n).astype(int),
        'CoapplicantIncome':  np.where(np.random.rand(n) < 0.45,
                                       np.random.lognormal(7.5, 0.8, n).astype(int), 0),
        'LoanAmount':         np.random.lognormal(4.9, 0.5, n).astype(int),
        'Loan_Amount_Term':   np.random.choice([360,180,120,60,480], n, p=[0.70,0.10,0.08,0.07,0.05]),
        'Credit_History':     np.random.choice([1.0, 0.0], n, p=[0.84, 0.16]),
        'Property_Area':      np.random.choice(['Urban','Semiurban','Rural'], n, p=[0.35,0.38,0.27]),
    }
    df = pd.DataFrame(data)
    # Introduce ~5% missing values
    for col in ['Gender','Married','Dependents','Self_Employed','LoanAmount','Loan_Amount_Term','Credit_History']:
        mask = np.random.rand(n) < 0.05
        df.loc[mask, col] = np.nan

    # Synthetic target
    score = (
        (df['Credit_History'].fillna(0) * 35) +
        (df['Education'] == 'Graduate').astype(int) * 10 +
        (df['Married'] == 'Yes').astype(int) * 8 +
        (df['Property_Area'] == 'Semiurban').astype(int) * 6 +
        (df['ApplicantIncome'] > 5000).astype(int) * 12 +
        (df['LoanAmount'].fillna(df['LoanAmount'].median()) < 150).astype(int) * 10 +
        np.random.randint(-10, 10, n)
    )
    df['Loan_Status'] = (score > 55).astype(int)
    return df

# ─────────────────────────────────────────────
# 2. EDA SUMMARY
# ─────────────────────────────────────────────
def eda_summary(df):
    print("=" * 55)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 55)
    print(f"\nShape       : {df.shape}")
    print(f"Duplicates  : {df.duplicated().sum()}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nLoan Approval Rate: {df['Loan_Status'].mean():.1%}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nNumerical Summary:\n{df.describe().T[['mean','std','min','max']].round(2)}")

# ─────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df):
    df = df.copy()

    # Handle missing values
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

    # Feature engineering
    df['Total_Income']         = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['Log_Total_Income']     = np.log1p(df['Total_Income'])
    df['Log_LoanAmount']       = np.log1p(df['LoanAmount'])
    df['EMI']                  = df['LoanAmount'] / df['Loan_Amount_Term'].replace(0, 1)
    df['Balance_Income']       = df['Total_Income'] - (df['EMI'] * 1000)
    df['Income_to_Loan_Ratio'] = df['Total_Income'] / (df['LoanAmount'] + 1)
    df['Debt_to_Income']       = (df['LoanAmount'] * 1000) / (df['Total_Income'] + 1)
    df['Dependents_num']       = df['Dependents'].replace({'3+': 3}).astype(float)
    df['Risk_Score']           = (
        df['Credit_History'] * 3 +
        (df['Income_to_Loan_Ratio'] > 10).astype(int) * 2 +
        (df['Debt_to_Income'] < 5).astype(int) * 2
    )

    # Encode categoricals
    le = LabelEncoder()
    cat_cols = ['Gender','Married','Education','Self_Employed','Property_Area']
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    feature_cols = [
        'Gender','Married','Dependents_num','Education','Self_Employed',
        'Log_Total_Income','Log_LoanAmount','Loan_Amount_Term','Credit_History',
        'Property_Area','EMI','Balance_Income','Income_to_Loan_Ratio',
        'Debt_to_Income','Risk_Score'
    ]
    X = df[feature_cols]
    y = df['Loan_Status']
    return X, y, feature_cols

# ─────────────────────────────────────────────
# 4. TRAIN & EVALUATE
# ─────────────────────────────────────────────
def train_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42),
        'XGBoost':             XGBClassifier(n_estimators=200, random_state=42,
                                             use_label_encoder=False, eval_metric='logloss',
                                             verbosity=0),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print("\n" + "=" * 55)
    print("  MODEL TRAINING & 5-FOLD CROSS VALIDATION")
    print("=" * 55)

    best_auc, best_model, best_name = 0, None, ""

    for name, model in models.items():
        Xtr = X_train_s if name == 'Logistic Regression' else X_train
        Xte = X_test_s  if name == 'Logistic Regression' else X_test

        cv_auc = cross_val_score(model, Xtr, y_train, cv=cv, scoring='roc_auc').mean()
        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)[:, 1]

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_prob)

        results[name] = dict(accuracy=acc, precision=prec, recall=rec, roc_auc=auc, cv_auc=cv_auc)

        print(f"\n{'─'*45}")
        print(f"  {name}")
        print(f"{'─'*45}")
        print(f"  CV ROC-AUC : {cv_auc:.4f}")
        print(f"  Accuracy   : {acc:.4f}")
        print(f"  Precision  : {prec:.4f}")
        print(f"  Recall     : {rec:.4f}")
        print(f"  ROC-AUC    : {auc:.4f}")

        if auc > best_auc:
            best_auc   = auc
            best_model = model
            best_name  = name
            best_scaler = scaler if name == 'Logistic Regression' else None

    print(f"\n✅ Best Model: {best_name} (ROC-AUC = {best_auc:.4f})")
    return best_model, best_name, best_scaler, results

# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    import os
    os.makedirs('models', exist_ok=True)

    df = generate_data()
    eda_summary(df)

    X, y, feature_cols = preprocess(df)
    best_model, best_name, best_scaler, results = train_evaluate(X, y)

    # Save artifacts
    joblib.dump(best_model,   'models/best_model.pkl')
    joblib.dump(feature_cols, 'models/feature_cols.pkl')
    if best_scaler:
        joblib.dump(best_scaler, 'models/scaler.pkl')

    print("\n💾 Model artifacts saved to /models/")
    print("\nAll results:")
    for m, r in results.items():
        print(f"  {m}: AUC={r['roc_auc']:.3f}")
