# 💳 Loan Approval Prediction System

A machine learning web application that predicts loan eligibility in real-time using scikit-learn and Flask.

## 🔧 Tech Stack
- **ML**: scikit-learn, XGBoost, pandas, numpy
- **Web**: Flask, HTML/CSS/JavaScript
- **Evaluation**: 5-fold cross-validation, ROC-AUC, Accuracy, Precision, Recall

## 📁 Project Structure
```
loan-approval-prediction/
├── app.py              # Flask web app
├── model.py            # EDA, training, evaluation
├── requirements.txt
├── models/             # Saved model artifacts (auto-created)
└── templates/
    └── index.html      # Frontend UI
```

## 🚀 Setup & Run

```bash
# 1. Clone & install
git clone https://github.com/YOUR_USERNAME/loan-approval-prediction.git
cd loan-approval-prediction
pip install -r requirements.txt

# 2. Train the model (saves to models/)
python model.py

# 3. Start Flask app
python app.py
```

Open http://localhost:5000

## 📊 Models Compared
| Model               | Metric   |
|---------------------|----------|
| Logistic Regression | ROC-AUC  |
| Random Forest       | ROC-AUC  |
| XGBoost             | ROC-AUC ✅ Best |

## ⚙️ Features Engineered
- `Total_Income`, `Log_Total_Income`, `Log_LoanAmount`
- `EMI`, `Balance_Income`
- `Income_to_Loan_Ratio`, `Debt_to_Income`
- `Risk_Score` (composite)
