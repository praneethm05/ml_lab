# Lab 3 — Simple Linear Regression: Quick Reference for Evaluation

**Dataset:** Student survey — CIA %, Attendance % → GPA prediction

---

## The Big Picture

Goal: predict a student's GPA using either their CIA % or their Attendance % via **Simple Linear Regression** (one input → one output).

Two approaches are compared:
1. **Scikit-learn** `LinearRegression` (library)
2. **Manual OLS** (Ordinary Least Squares, coded from scratch)

---

## Part A — Data Collection & Preprocessing

### Step 1 — Imports
```python
import pandas as pd, numpy as np, matplotlib.pyplot as plt, pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```
- `pandas` → tabular data handling
- `numpy` → array math
- `matplotlib` → plots
- `pickle` → save/load model weights
- `sklearn` → train/test split, model, metrics

---

### Step 2 — Load Dataset
```python
df = pd.read_csv('student_survey.csv')
df.columns = [c.strip() for c in df.columns]
```
`strip()` removes accidental whitespace from column names — prevents `KeyError` when accessing columns by name.

---

### Step 3 — `df.head()`
Shows first 5 rows. Sanity check that data loaded correctly.

---

### Step 4 — `df.shape`
54 rows × 15 columns. Tells you total sample size before cleaning.

---

### Step 5 — `df.isnull().sum()`
Counts missing values per column. Needed before deciding whether to drop or fill.

---

### Step 6 — Select Columns & Convert to Numeric
```python
df[cia_col] = pd.to_numeric(df[cia_col], errors='coerce')
```
- Some attendance entries had text like `"Option 1"` instead of a number.
- `errors='coerce'` turns invalid strings → `NaN` (instead of crashing).
- Without this step, the column stays as `object` dtype and math operations fail.

---

### Step 7 — Drop Nulls & Invalid Ranges
```python
df = df.dropna(subset=[cia_col, gpa_col, att_col])
df = df[(df[gpa_col] >= 0) & (df[gpa_col] <= 10)]
df = df[(df[cia_col] >= 0) & (df[cia_col] <= 100)]
df = df[(df[att_col] >= 0) & (df[att_col] <= 100)]
```
- GPA valid range: 0–10 (some students entered 90, which is wrong for a 10-point scale).
- CIA/Attendance valid range: 0–100 (some entered decimals like 0.7 instead of 70).
- Filtering keeps only realistic values.

---

### Step 8 — Drop Duplicates
```python
df = df.drop_duplicates()
```
Removes identical rows — prevents model from being biased toward repeated entries.

---

### Step 9 — `df.describe()`
Statistical summary: mean, std, min, max, quartiles.
- Avg CIA ≈ 57.75%, Avg GPA ≈ 3.75, most students above 87% attendance.

---

### Step 10 — Extract Arrays
```python
X_cia = df[cia_col].values   # 1D numpy array
X_att = df[att_col].values
y     = df[gpa_col].values
```
`.values` converts DataFrame column → numpy array for sklearn.

---

## Part B — Scikit-learn Linear Regression

### Experiment 1: CIA % → GPA

#### Scatter Plot
```python
plt.scatter(X_cia, y)
```
Visual check: are CIA % and GPA linearly related? (Spoiler: they're not, data is scattered.)

---

#### Train-Test Split
```python
X_cia_2d = X_cia.reshape(-1, 1)  # sklearn expects 2D: (n_samples, n_features)
X_cia_train, X_cia_test, y_cia_train, y_cia_test = train_test_split(
    X_cia_2d, y, test_size=0.2, random_state=2
)
```
- `reshape(-1, 1)`: converts 1D `[a, b, c]` → 2D `[[a], [b], [c]]` — sklearn requirement.
- `test_size=0.2`: 80% train, 20% test.
- `random_state=2`: fixed seed → reproducible split every run.

---

#### Train Model
```python
lr_cia = LinearRegression()
lr_cia.fit(X_cia_train, y_cia_train)
print(lr_cia.coef_[0])      # slope m
print(lr_cia.intercept_)    # intercept b
```
Finds the best-fit line: `GPA = m × CIA% + b` by minimizing sum of squared errors (OLS internally).

---

#### Predict
```python
y_cia_pred = lr_cia.predict(X_cia_test)
lr_cia.predict([[75]])  # predict for a single CIA% value
```
Applies the learned equation to unseen test data.

---

#### Plot Regression Line
```python
plt.scatter(X_cia, y, color='blue')
plt.plot(X_cia_train, lr_cia.predict(X_cia_train), color='red')
```
Blue dots = actual data, red line = learned regression line.

---

#### Evaluation Metrics
```python
mean_absolute_error(y_cia_test, y_cia_pred)   # MAE
mean_squared_error(y_cia_test, y_cia_pred)    # MSE
r2_score(y_cia_test, y_cia_pred)              # R²
```

| Metric | What it means |
|--------|--------------|
| **MAE** | Average absolute difference between actual and predicted GPA |
| **MSE** | Average squared difference — penalizes large errors more |
| **R²** | How much variance the model explains. 1 = perfect, 0 = predicts mean, **negative = worse than predicting mean** |

**Result:** R² is **negative** for CIA% → model worse than just guessing the average GPA. Data is too scattered for a linear fit.

---

### Experiment 2: Attendance % → GPA

Same pipeline as Experiment 1, with `X_att` instead of `X_cia`.

**Result:** R² ≈ 0 — almost no linear relationship between attendance and GPA in this data. Slightly better than CIA% but still very weak.

---

## Part C — Manual OLS (From Scratch)

### OLS Formulas

$$m = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sum (x_i - \bar{x})^2}$$

$$b = \bar{y} - m \cdot \bar{x}$$

- **m (slope):** how much GPA changes per 1% change in CIA/Attendance
- **b (intercept):** predicted GPA when the input = 0

### Custom `LR` Class
```python
class LR:
    def fit(self, X_train, y_train):
        num, den = 0, 0
        for i in range(X_train.shape[0]):
            num += (X_train[i] - X_train.mean()) * (y_train[i] - y_train.mean())
            den += (X_train[i] - X_train.mean()) ** 2
        self.m = num / den
        self.b = y_train.mean() - (self.m * X_train.mean())

    def predict(self, X_test):
        return self.m * X_test + self.b
```
- Loops over all samples, computes numerator (covariance) and denominator (variance of X).
- Derives slope `m` then intercept `b`.
- Trained on **full dataset** (not split), unlike sklearn which used 80% train set.

---

## Part D — Sklearn vs OLS Comparison

```python
print("Sklearn  Slope:", round(lr_cia.coef_[0], 4))
print("OLS      Slope:", round(lr_cia_ols.m, 4))
```

**Why slopes differ slightly:** Sklearn trained on 80% of data; OLS trained on 100%. Same algorithm, different data → slightly different parameters. If trained on identical data, results would be **exactly the same** (sklearn `LinearRegression` uses OLS internally).

**Key takeaway:** Both give almost identical predictions — difference column values are near zero.

---

## Part E — Pickle: Save & Load Model

### Save
```python
parameters = {
    'exp1_cia_gpa': {'slope': lr_cia.coef_[0], 'intercept': lr_cia.intercept_},
    'exp2_att_gpa': {'slope': lr_att.coef_[0], 'intercept': lr_att.intercept_}
}
with open('linear_regression_weights.pkl', 'wb') as f:
    pickle.dump(parameters, f)
```
`'wb'` = write binary. Serializes the Python dict to disk.

### Load & Use
```python
with open('linear_regression_weights.pkl', 'rb') as f:
    loaded_params = pickle.load(f)

m = loaded_params['exp1_cia_gpa']['slope']
b = loaded_params['exp1_cia_gpa']['intercept']
predicted_gpa = m * 75 + b   # predict for CIA% = 75
```
`'rb'` = read binary. Restores the dict without retraining.

**Why pickle?** Real-world models take hours to train. Pickle lets you save weights once and reuse them for inference anytime.

---

## Final Conclusions

| Experiment | Predictor | R² | Verdict |
|------------|-----------|-----|---------|
| 1 | CIA % → GPA | Negative | Poor fit — data too scattered |
| 2 | Attendance % → GPA | ≈ 0 | Near-zero relationship |

- Neither CIA% nor Attendance% alone predicts GPA reliably in this dataset.
- Small sample (45 rows after cleaning) + inconsistent survey responses = noisy data.
- GPA depends on many factors — single-variable linear regression is insufficient here.
- Sklearn and manual OLS give the same result when trained on the same data (both use OLS math).
- Pickle = standard way to persist and reuse trained model parameters.

---

## Key Concepts to Know for Evaluation

| Concept | One-line answer |
|---------|----------------|
| Simple Linear Regression | One input → one output, fits line `y = mx + b` |
| OLS | Method to find m and b by minimizing sum of squared residuals |
| `reshape(-1, 1)` | Converts 1D array to 2D column — sklearn needs 2D input |
| `errors='coerce'` | Converts non-numeric strings to NaN during type conversion |
| `test_size=0.2` | 80/20 train-test split |
| `random_state` | Fixes randomness for reproducibility |
| R² < 0 | Model worse than predicting the mean — very bad fit |
| R² = 1 | Perfect prediction |
| MAE vs MSE | MAE = average error; MSE penalizes outliers more (squares errors) |
| Pickle | Serialize Python objects to binary file for persistence |
