# Census Income Prediction: Logistic Regression and Model Selection

Predicting whether a person earns more than $50K a year from 1994 U.S. Census data. The project uses logistic regression, handles class imbalance, tunes hyperparameters with GridSearchCV, interprets the coefficients, and considers fairness.

Built as my capstone for the **Break Through Tech AI Machine Learning Foundations** course (Cornell Tech).

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

---

## 🎯 Highlights

- **ROC-AUC of 0.91 and recall of 0.85** on the >$50K class, using a held-out test set of 6,513 people
- **Handled class imbalance** (76% / 24%) with `class_weight='balanced'` and chose **F1 over accuracy**. Predicting "≤$50K" for everyone would score 76% accuracy while finding no one in the minority class.
- **Searched 12 hyperparameter combinations** with 5-fold GridSearchCV. The search showed that class weighting matters far more than regularization strength `C`.
- **Interpreted every coefficient as an odds ratio** and set protected attributes aside to audit the model's fairness

---

## 🏗️ Problem

**Task:** binary classification. The label is `income_binary`, where 1 means income above $50K.
**Use case:** ranking an intake queue, as a community lender might do. A model that outputs probabilities can tell a 0.51 from a 0.95, which matters more here than a hard yes/no label.

---

## 📊 Data

**Dataset:** UCI Adult / Census Income, with 32,561 rows and 15 columns.

**Preparation decisions:**
| Step | Decision | Why |
|---|---|---|
| Label | Mapped `>50K` to 1 | F1 is measured on the positive class, so the minority class has to be 1 |
| Missing `workclass` / `occupation` (~1,840 rows) | Filled with an `Unknown` category | Dropping these rows would delete unemployed and retired people. Missingness is itself a signal. |
| Missing `age` / `hours-per-week` | Filled with the median | Robust to skewed distributions |
| Dropped `education`, `fnlwgt`, `native-country` | | `education` duplicates `education-num`, `fnlwgt` is a sampling weight, and `native-country` has 41 sparse levels |
| `sex_selfID`, `race` | Removed from the features but kept for the fairness audit | These are protected attributes |
| Encoding and scaling | One-hot encoding (`drop_first`), then `StandardScaler` | The final feature matrix has 38 columns |

The train/test split is **80/20 and stratified**, giving 26,048 training rows and 6,513 test rows.

---

## 🧠 Model Development

1. **Baseline:** `LogisticRegression(class_weight='balanced')`
2. **Model selection:** `GridSearchCV` over `C ∈ {0.001, 0.01, 0.1, 1, 10, 100}` × `class_weight ∈ {None, 'balanced'}`, with 5-fold cross-validation, **scored on F1**
3. **Final model:** refit using the best parameters, `C=1` and `class_weight='balanced'`

---

## 📈 Results

| Metric (test set) | Score |
|---|---|
| Accuracy | 0.808 |
| **F1 (>$50K)** | **0.680** |
| ROC-AUC | 0.908 |
| Recall (>$50K) | 0.85 |
| Precision (>$50K) | 0.57 |

**Key findings**
- **Class weighting mattered most.** Every `balanced` setting beat every unweighted one in cross-validation, with an F1 of about 0.677 versus about 0.655. Changing `C` barely moved the score, because 26K rows and 38 features leave regularization little to do.
- **The choice of metric changes which model wins.** A search scored on accuracy would have chosen `class_weight=None`. That model looks better on paper but is worse at finding the people who earn above $50K.
- **The strongest predictors** are being married (odds ratio **3.0**), capital gains (2.3), years of education (2.1), hours per week (1.6) and age (1.5). The large marriage effect is a warning: the model is learning household structure, not just individual earning power.

---

## 👩🏽‍💻 How to Run

```bash
git clone https://github.com/sameeksha-mehrotra/logistic-regression-model-selection.git
cd logistic-regression-model-selection
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook
```
Open the notebook and run all cells. It expects the dataset at `data_capstone/censusData.csv`. You can download the [UCI Adult dataset](https://archive.ics.uci.edu/dataset/2/adult) if you don't have it.

---

## ⚖️ Ethical Considerations

- The data comes from **1994**. Wages, occupations and what $50K means have all changed since then, so this model shouldn't be used for real decisions without retraining on current data.
- I removed protected attributes from the features, but proxies like marital status and relationship still carry demographic signal. That's why I kept `sex_selfID` and `race` aside to audit error rates by group.

---

## 🚀 Next Steps

- Tune the decision threshold to balance precision and recall for the intake-queue use case
- Finish the fairness audit by comparing false-positive and false-negative rates across groups
- Compare against a neural network and a gradient-boosted tree model

---

**Sameeksha Mehrotra** · [LinkedIn](https://www.linkedin.com/in/sameeksha-mehrotra) · [GitHub](https://github.com/sameeksha-mehrotra)
