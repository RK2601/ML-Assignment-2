import pandas as pd
import joblib
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

test = pd.read_csv("test_data.csv")
X = test.drop(columns=["diagnosis"])
y = test["diagnosis"]

MODELS = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib",
}

rows = []
for name, path in MODELS.items():
    model = joblib.load(path)
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]   # probability of malignant (class 1)
    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, proba),
        "Precision": precision_score(y, pred),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "MCC": matthews_corrcoef(y, pred),
    })

table = pd.DataFrame(rows).round(3)
print(table.to_string(index=False))
table.to_csv("results.csv", index=False)