# Breast Cancer Classification: Five Models Compared

## 1. Problem Statement
Classify breast mass fine-needle aspirate (FNA) samples as malignant or benign, using measurements of cell nuclei computed from digitized images.

## 2. Dataset Description
- **Source:** UCI Machine Learning Repository, Breast Cancer Wisconsin (Diagnostic)
- **Instances:** 569
- **Features:** 30 numeric features (mean, standard error, and worst value of 10 nucleus measurements)
- **Target:** `diagnosis` (M = malignant, B = benign), encoded as 1/0
- **Class balance:** 212 malignant, 357 benign
- **Data checks:** no missing values, no duplicate rows, no duplicate IDs, all labels valid, all features numeric
- **Preprocessing:** dropped the `id` column; 80/20 stratified train/test split (random_state=42); features standardized inside each pipeline, fit on training data only

## 3. GitHub Repository
https://github.com/RK2601/ML-Assignment-2

## 4. Models Used

### Comparison Table (test set, 114 rows)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.965 | 0.996 | 0.975 | 0.929 | 0.951 | 0.925 |
| Decision Tree | 0.956 | 0.945 | 0.974 | 0.905 | 0.938 | 0.906 |
| KNN | 0.956 | 0.982 | 0.974 | 0.905 | 0.938 | 0.906 |
| Naive Bayes | 0.939 | 0.993 | 1.000 | 0.833 | 0.909 | 0.871 |
| Support Vector Machine | 0.974 | 0.995 | 1.000 | 0.929 | 0.963 | 0.944 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | *(write your own observation)* |
| Decision Tree | *(write your own observation)* |
| KNN | *(write your own observation)* |
| Naive Bayes | *(write your own observation)* |
| Support Vector Machine | *(write your own observation)* |

**Overall winner for this dataset:** *(write your choice and reason)*