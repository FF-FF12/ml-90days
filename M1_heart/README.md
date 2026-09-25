# Heart Disease Prediction with Machine Learning

Predicting coronary heart disease from clinical indicators, and identifying which indicators carry the strongest warning signal.

> **Disclaimer**: This is a learning project. The model is NOT a diagnostic tool and must not be used for any clinical decision.

---

## 1. What this project does

Given 13 clinical indicators of a patient (age, chest pain type, max heart rate, ST depression, etc.), predict whether the patient has coronary heart disease.

Beyond prediction, the project answers a second question: **which indicators matter most?** — because in a clinical context, a model that cannot explain itself has limited value.

---

## 2. Data

| Item | Detail |
|---|---|
| Source | UCI Machine Learning Repository — Heart Disease (Cleveland) |
| Raw samples | 303 |
| After cleaning | **297** (6 records with missing `ca` / `thal` dropped) |
| Features | 13 |
| Target | `target`: 0 = no disease (160), 1 = disease (137) |
| Missing values | `ca` (4 records), `thal` (2 records) |

**Feature description**

| Feature | Meaning | Type |
|---|---|---|
| `age` | Age | numeric |
| `sex` | 1 = male, 0 = female | categorical |
| `cp` | Chest pain type (1–4) | categorical |
| `trestbps` | Resting blood pressure (mm Hg) | numeric |
| `chol` | Serum cholesterol (mg/dl) | numeric |
| `fbs` | Fasting blood sugar > 120 mg/dl | categorical |
| `restecg` | Resting ECG results (0–2) | categorical |
| `thalach` | Maximum heart rate achieved | numeric |
| `exang` | Exercise-induced angina | categorical |
| `oldpeak` | ST depression induced by exercise | numeric |
| `slope` | Slope of peak exercise ST segment | categorical |
| `ca` | Number of major vessels (0–3) | numeric |
| `thal` | Thalassemia (3 = normal, 6 = fixed, 7 = reversible defect) | categorical |

---

## 3. Method

### 3.1 Why recall, not accuracy

In this dataset, **a naive model that predicts "no disease" for everyone already achieves ~54% accuracy.** Accuracy is therefore useless as a metric.

In cardiovascular screening, **a false negative (missing a patient with heart disease) costs far more than a false positive (sending a healthy person for further tests).** The primary metric is therefore **recall (sensitivity)**: of all patients who truly have the disease, how many does the model catch?

> For the same reason, `pos_label=1` is explicitly specified throughout — because `target=1` encodes "has disease" in this dataset. Default settings happen to match here, but relying on that is how evaluation quietly goes wrong.

### 3.2 Preprocessing

- **One-hot encoding** for 4 categorical features (`cp`, `restecg`, `slope`, `thal`) — 13 columns become **18**.
  These codes are nominal labels, not magnitudes. Feeding them in raw would tell the model that "chest pain type 4 is 4 times type 1", which is false.
  `drop_first=True` is applied to avoid multicollinearity.
- **Stratified split**: 237 train / 60 test, with `stratify=y` so both sets keep the same disease ratio (train 0.460, test 0.467). With only 297 samples, an unstratified split easily produces a distorted test set.

### 3.3 Models

Logistic Regression, Decision Tree, Random Forest — all evaluated with **5-fold cross-validation** on the training set, then finally on the held-out test set.

---

## 4. Results

### Model comparison (metric = recall)

| Model | CV Recall | Test Recall |
|---|---|---|
| Logistic Regression | 0.769 | **0.821** |
| Decision Tree | 0.770 | 0.607 |
| Random Forest | 0.760 | 0.786 |

**Key observation**: all three models score nearly identically under cross-validation (0.76–0.77), yet the Decision Tree **drops to 0.607 on the test set** — a 16-point fall. This is textbook overfitting: a single tree memorises noise in the training data and fails to generalise.

Random Forest recovers stability by averaging 300 trees. With only 237 training samples, **simpler models generalise better** — the plain Logistic Regression actually performed best on unseen data.

### Confusion matrix (Random Forest, test set)

|  | Predicted: No disease | Predicted: Disease |
|---|---|---|
| **Actual: No disease** | 28 (TN) | 4 (FP) |
| **Actual: Disease** | **6 (FN)** | 22 (TP) |

- Recall = 22 / (22 + 6) = **0.786** (verified by hand-calculation)
- Accuracy = 0.83 — but note that **false negatives (6) exceed false positives (4)**

**This is the most important finding of the project.** The model's errors lean toward the dangerous side: it misses more genuine patients than it falsely alarms. In a screening context, a missed case is far more costly than an unnecessary follow-up exam. Accuracy of 0.83 hides this asymmetry entirely.

### Feature importance (top 5)

| Rank | Feature | Importance | Clinical meaning |
|---|---|---|---|
| 1 | `oldpeak` | 0.1177 | ST depression induced by exercise |
| 2 | `thal_7.0` | 0.1165 | Reversible perfusion defect |
| 3 | `thalach` | 0.1078 | Maximum heart rate achieved |
| 4 | `cp_4.0` | 0.1040 | Asymptomatic chest pain |
| 5 | `ca` | 0.1017 | Number of major vessels coloured by fluoroscopy |

**The model's ranking agrees with clinical knowledge**, and also with what was observed independently during exploratory analysis (before any model was trained):

- `oldpeak` and `thal_7.0` are direct markers of myocardial ischaemia.
- `thalach` ranked third — matching the EDA finding that the disease group's max heart rate distribution peaks notably lower (~140 vs ~160).
- `cp_4.0` ranked fourth, corresponding to **asymptomatic** patients carrying the highest disease proportion. This matches the clinically recognised danger of silent myocardial ischaemia: no obvious symptoms, easily overlooked, worse outcome.

---

## 5. Conclusions and limitations

**Conclusions**

1. On this dataset, ML models reach ~0.79 recall on unseen data, but none is reliable enough for real screening — too many patients are missed.
2. With limited data (237 training samples), model complexity is a liability. Simple models and ensembling both beat a single overfitted tree.
3. The features the model relies on are largely consistent with clinical understanding, which supports that it learned real signal rather than noise.

**Limitations**

- **Sample size is small** (297 records). Conclusions are indicative, not robust; feature importance ranking may shift with resampling.
- Feature importance reflects **statistical association, not causation**. No causal claim is made.
- Records with missing values were dropped rather than imputed, which is a simplification.
- No hyperparameter tuning or threshold optimisation was performed in this iteration.

**Next steps**: increase recall by lowering the decision threshold or applying class weighting; try imputation instead of dropping; validate on a larger cohort.

---

## Repository structure

```
M1_heart/
├── README.md
├── heart_disease_clean.csv      # cleaned dataset used for modelling
├── heart_disease_raw.csv        # raw version with missing values
└── m1_heart.ipynb               # full analysis notebook
```

## Environment

Python 3 · scikit-learn · pandas · numpy · matplotlib

Run with Jupyter Notebook from this directory:

```
jupyter notebook
```
