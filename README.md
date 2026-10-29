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
- **Preprocessing:** dropped the `id` column; 80/20 stratified train/test split (random_state=42); features standardized inside the Logistic Regression, KNN, and SVM pipelines, fit on training data only

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

| ML Model Name | Observations |
|---|---|
| **Logistic Regression** | Logistic Regression achieved an accuracy of **0.965**, an AUC of **0.996 (highest)**, recall of **0.929**, and MCC of **0.925**. Its high AUC indicates excellent class discrimination across different classification thresholds, while its accuracy is slightly lower than SVM's **0.974**, suggesting SVM makes fewer classification errors at the selected thresholds. |
| **Decision Tree** | The Decision Tree achieved an accuracy of **0.956**, precision of **0.974**, recall of **0.905**, and an AUC of **0.945 (lowest)**. Recall is its weakest metric, indicating that it misses more actual positive cases; additionally, its training accuracy of **0.991** compared with its test accuracy of **0.956** indicates a possible overfitting issue. |
| **KNN** | KNN achieved an accuracy of **0.956**, an AUC of **0.982**, recall of **0.905**, and an F1-score of **0.938**. Compared with the Decision Tree, KNN has substantially better AUC, indicating stronger overall discrimination, although both models have identical F1 and MCC scores. |
| **Naive Bayes** | Naive Bayes achieved perfect precision of **1.000**, but its recall of **0.833 was the lowest**, with an F1-score of **0.909**. This means every case predicted as malignant was malignant in the test set, but approximately **16.7% of actual malignant cases were missed**, making it less suitable when detecting every malignant case is critical. |
| **SVM** | SVM achieved the highest accuracy of **0.974**, an F1-score of **0.963**, an MCC of **0.944**, and an AUC of **0.995**. Its high precision of **1.000** and recall of **0.929**, combined with the highest accuracy and MCC, make it the strongest overall performer. |
| **Overall Winner: SVM** | SVM is the overall winner because it has the **highest accuracy (0.974)** and **highest MCC (0.944)**, alongside an F1-score of **0.963** and AUC of **0.995**. Although Logistic Regression achieves a slightly higher AUC of **0.996**, SVM performs better on accuracy, F1, and MCC; however, because the test set contains only **114 rows**, cross-validation or a larger independent test set is needed to confirm the ranking. |