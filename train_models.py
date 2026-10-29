import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# 1. Load the file (it has no header row, so we name the columns)
COLS = ["id", "diagnosis",
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean",
        "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
        "compactness_se", "concavity_se", "concave_points_se", "symmetry_se",
        "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
        "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst",
        "fractal_dimension_worst"]
FEATURES = COLS[2:]

df = pd.read_csv("data.csv", header=None, names=COLS)

# 2. Data checks (script stops here if something is wrong)
print("Shape:", df.shape)
assert df.isna().sum().sum() == 0, "Missing values found"
assert df.duplicated().sum() == 0, "Duplicate rows found"
assert df["id"].is_unique, "Duplicate IDs found"
assert set(df["diagnosis"].unique()) <= {"M", "B"}, "Unexpected diagnosis labels"
assert all(pd.api.types.is_numeric_dtype(df[c]) for c in FEATURES), "Non-numeric feature"
print("Class counts:\n", df["diagnosis"].value_counts())
print("All data checks passed.")

# 3. Cleaning
df = df.drop(columns=["id"])
df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})   # malignant = 1 (positive class)

X = df[FEATURES]
y = df["diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# 4. Models (scaling lives inside the pipeline, so it is fit on training data only)
models = {
    "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "decision_tree": DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=42),
    "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "naive_bayes": GaussianNB(),
    "svm": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42)),
}

# 5. Train and save
os.makedirs("model", exist_ok=True)
for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"model/{name}.joblib")
    print(f"{name}: train accuracy = {model.score(X_train, y_train):.3f}")

# 6. Save the held-out test set for the Streamlit app
test = X_test.copy()
test["diagnosis"] = y_test
test.to_csv("test_data.csv", index=False)
print("Saved models to model/ and test_data.csv")