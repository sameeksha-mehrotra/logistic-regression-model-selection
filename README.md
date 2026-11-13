# Airbnb Superhost Prediction: Logistic Regression and Model Selection

Predicting whether a New York City Airbnb host is a **Superhost** from listing and host data. A logistic regression classifier was tuned through model selection over the regularization strength `C`, and the best model was saved for reuse.

Built in the **Break Through Tech AI Machine Learning Foundations** course (Cornell Tech).

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)

---

## 🎯 Highlights

- **Binary classification** on 28,022 NYC Airbnb listings with 49 engineered features
- **Model selection** over the regularization hyperparameter `C`. The selected model uses `C = 100` with L2 penalty and the `lbfgs` solver.
- **Saved model** (`model_best.pkl`) that can be loaded and used for predictions without retraining
- **ROC-AUC of 0.82**, compared with a 75.7% majority-class baseline

---

## 📁 What's in This Repo

| File | Description |
|---|---|
| `airbnbData_train.csv` | Cleaned, model-ready dataset: 28,022 listings × 50 columns (1 label + 49 features), with no missing values |
| `model_best.pkl` | The best logistic regression model found during model selection, saved with `pickle` |

---

## 📊 Data

**Label:** `host_is_superhost`, which is True for 24.3% of listings and False for 75.7%

**Features (49):**
- **Host:** response and acceptance rate, listing counts, profile picture, identity verification, number of verifications
- **Listing:** guests accommodated, bedrooms, bathrooms, beds, price, minimum and maximum nights, instant bookable
- **Availability:** next 30, 60, 90 and 365 days
- **Reviews:** total reviews, reviews in the last 12 months and last 30 days, reviews per month, and six review-score categories (overall, cleanliness, check-in, communication, location, value)
- **One-hot encoded categories:** borough (Bronx, Brooklyn, Manhattan, Queens, Staten Island) and room type (entire home, hotel room, private room, shared room)

**Preprocessing already applied:** missing values filled in, booleans converted to 0/1, and categorical columns one-hot encoded

---

## 🧠 Model

| Setting | Value |
|---|---|
| Algorithm | `sklearn.linear_model.LogisticRegression` |
| Regularization | L2, **C = 100** (chosen by model selection) |
| Solver | `lbfgs`, `max_iter = 1000` |
| Features | 49 |

**Why model selection matters:** `C` controls regularization strength. A small `C` simplifies the model and can underfit, while a large `C` fits the training data more closely. Trying a range of `C` values and keeping the best one avoids guessing at this tradeoff.

---

## 📈 Results

These are scores of the saved model on the included dataset:

| Metric | Score |
|---|---|
| ROC-AUC | 0.82 |
| Accuracy | 0.81 (majority-class baseline: 0.757) |
| F1 (Superhost class) | 0.48 |

**Takeaway:** the model ranks hosts well (AUC 0.82), but at the default 0.5 threshold it misses many Superhosts (F1 0.48). That's typical with imbalanced labels, and it points to the next steps below.

---

## 👩🏽‍💻 How to Use

```bash
git clone https://github.com/sameeksha-mehrotra/logistic-regression-model-selection.git
cd logistic-regression-model-selection
pip install pandas scikit-learn
```

```python
import pickle
import pandas as pd
from sklearn.metrics import roc_auc_score

# Load the data and the saved model
df = pd.read_csv("airbnbData_train.csv")
with open("model_best.pkl", "rb") as f:
    model = pickle.load(f)

y = df["host_is_superhost"].astype(int)
X = df[model.feature_names_in_]

# Predict the probability that each host is a Superhost
probs = model.predict_proba(X)[:, 1]
print("ROC-AUC:", round(roc_auc_score(y, probs), 3))
```

> Only load pickle files you trust. Unpickling can run code.

---

## 🚀 Next Steps

- **Scale features** with `StandardScaler` before training. Features like `price` and `maximum_nights` have very different ranges, which affects both regularization and how coefficients can be interpreted.
- **Handle class imbalance** with `class_weight='balanced'` and tune the decision threshold to improve recall on Superhosts
- **Use F1 or ROC-AUC as the model selection metric** through `GridSearchCV` with cross-validation
- **Compare** against decision trees, random forests and gradient boosting

---

**Sameeksha Mehrotra** · [LinkedIn](https://www.linkedin.com/in/sameeksha-mehrotra) · [GitHub](https://github.com/sameeksha-mehrotra)
