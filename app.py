import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)

TARGET = "diagnosis"
MODELS = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib",
}

@st.cache_resource
def load_model(path):
    return joblib.load(path)

def compute_metrics(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, proba),
        "Precision": precision_score(y, pred),
        "Recall": recall_score(y, pred),
        "F1": f1_score(y, pred),
        "MCC": matthews_corrcoef(y, pred),
    }

st.title("Breast Tumor Diagnosis Model Explorer")
st.write(
    "This app compares classifiers on the Wisconsin Diagnostic Breast Cancer "
    "dataset. Upload `test_data.csv` with the 30 numeric nucleus-measurement "
    "features and a `diagnosis` column (0 = benign, 1 = malignant), then pick "
    "a model to inspect its results."
)

uploaded = st.file_uploader(
    "Choose a labeled breast-cancer test CSV",
    type="csv",
    help="Include the model's 30 numeric feature columns and a diagnosis column "
    "encoded as 0 (benign) or 1 (malignant).",
)
if uploaded is None:
    st.info(
        "Start with `test_data.csv`, or provide another labeled CSV containing "
        "the 30 nucleus measurements and `diagnosis` (0 = benign, 1 = malignant)."
    )
    st.stop()

data = pd.read_csv(uploaded)
if TARGET not in data.columns:
    st.error(f"Uploaded file must contain the '{TARGET}' column.")
    st.stop()
X, y = data.drop(columns=[TARGET]), data[TARGET]

choice = st.sidebar.selectbox("Model to inspect", list(MODELS))
model = load_model(MODELS[choice])

st.header(f"{choice}: performance on this file")
metrics = compute_metrics(model, X, y)
metric_items = list(metrics.items())
for start in range(0, len(metric_items), 3):
    metric_columns = st.columns(3)
    for column, (name, value) in zip(
        metric_columns, metric_items[start:start + 3]
    ):
        column.metric(name, f"{value:.3f}")

matrix_tab, report_tab, comparison_tab = st.tabs(
    ["Prediction counts", "Class-by-class scores", "Compare all models"]
)
with matrix_tab:
    st.subheader("How often did predictions match each diagnosis?")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix(y, model.predict(X))).plot(ax=ax)
    st.pyplot(fig)

with report_tab:
    st.subheader("Precision, recall, and F1 for each diagnosis")
    st.text(classification_report(y, model.predict(X)))

with comparison_tab:
    st.subheader("Model scores on the uploaded file")
    rows = {
        name: compute_metrics(load_model(path), X, y)
        for name, path in MODELS.items()
    }
    comparison = pd.DataFrame(rows).T
    best_model = comparison["MCC"].idxmax()
    st.success(
        f"{best_model} has the highest MCC on this file "
        f"({comparison.loc[best_model, 'MCC']:.3f})."
    )
    st.dataframe(comparison.round(3))