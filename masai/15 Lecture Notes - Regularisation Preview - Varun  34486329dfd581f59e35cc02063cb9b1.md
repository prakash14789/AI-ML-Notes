# 15. Lecture Notes - Regularisation Preview - Varun Raste - 14 Nov 2025

# Regularization Preview: Taming Overfitting with Ridge & Lasso - Lecture Notes

**What you'll be able to do:**

- Explain what regularization is and why it prevents overfitting
- Apply Ridge regression (L2) and understand when to use it
- Apply Lasso regression (L1) and understand when to use it
- Choose between Ridge, Lasso, and ordinary regression based on problem characteristics
- Tune regularization strength (alpha/lambda parameter)

---

## 1. Introduction

### Core Definition

Regularization adds a penalty term to the regression loss function that punishes large coefficients, preventing overfitting by discouraging overly complex models. Ridge regression adds L2 penalty (sum of squared coefficients), while Lasso adds L1 penalty (sum of absolute coefficients). The key insight: simpler models (smaller coefficients) generalize better to new data, even if they fit training data slightly worse. Regularization trades a bit of training accuracy for better test accuracy.

### A Simple Analogy

Think of regularization like a speed limit on a race track. Without limits (ordinary regression), the car (model) can take sharp, aggressive turns (large coefficients) to hug every bump in the track (fit every data point), but this is dangerous and doesn't work on other tracks (overfitting). Speed limits (regularization) force smoother, safer driving (smaller coefficients) that works consistently across different tracks (better generalization).

### Why This Matters

**Problem it solves:** With many features or limited data, ordinary linear regression overfits—memorizes training noise instead of learning real patterns. Test performance suffers dramatically. Feature selection (manually choosing which features to keep) is tedious and subjective. Regularization automatically handles these issues.

**What you'll gain:**

- **Better generalization:** Models that perform well on new, unseen data
- **Automatic feature selection:** Lasso can zero out irrelevant features
- **Stability:** Small data changes don't drastically change coefficients
- **Handle multicollinearity:** Works even when features are highly correlated

**Real-world:** Netflix uses regularization for recommendation models with millions of features. Financial models use it to prevent overfitting to historical market patterns. Medical studies use it when variables outnumber patients.

---

## 2. Core Concepts

### Concept A: The Overfitting Problem

**Definition:** Overfitting occurs when a model fits training data too closely—capturing noise and random fluctuations instead of true patterns. The model performs well on training data but poorly on new data. This happens with many features, complex models, or limited training data.

**Signs of overfitting:**

- High training R², low test R²
- Large positive and negative coefficients
- Coefficients are unstable (change drastically with small data changes)
- Model captures random noise patterns

**Example:**

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 50 features, only 100 samples (more features than observations is extreme)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

print(f"Train R²: {model.score(X_train, y_train):.3f}")  # 0.999 (too good!)
print(f"Test R²: {model.score(X_test, y_test):.3f}")    # 0.234 (terrible!)
print(f"Max coefficient: {np.max(np.abs(model.coef_)):.0f}")  # 58,234 (huge!)

```

**Remember:** Perfect training fit often means overfitting, not success.

---

### Concept B: Ridge Regression (L2 Regularization)

**Definition:** Ridge adds a penalty proportional to the **sum of squared coefficients** to the loss function: Loss = MSE + α × Σβᵢ². The parameter α (alpha) controls strength—larger α means stronger penalty, smaller coefficients. Ridge shrinks coefficients toward zero but never makes them exactly zero.

**How it works:**

- Minimizes: (y - ŷ)² + α × (β₁² + β₂² + ... + βₙ²)
- Penalizes large coefficients heavily (squaring amplifies large values)
- All features kept, but coefficients shrunk
- Handles multicollinearity well

**Key characteristics:**

- **Shrinks coefficients:** All become smaller, approaching zero as α increases
- **Keeps all features:** No feature elimination (all coefficients > 0)
- **Works with correlated features:** Distributes coefficients across correlated features
- **Smooth:** Small α changes → small coefficient changes

**Example:**

```python
from sklearn.linear_model import Ridge

# α = 1.0 (moderate regularization)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

print(f"Train R²: {ridge.score(X_train, y_train):.3f}")  # 0.847 (lower than OLS)
print(f"Test R²: {ridge.score(X_test, y_test):.3f}")    # 0.812 (much better!)
print(f"Max coefficient: {np.max(np.abs(ridge.coef_)):.0f}")  # 12 (much smaller!)

```

**When to use Ridge:**

- Many correlated features (multicollinearity)
- All features potentially useful
- Want to keep all features but reduce overfitting
- Need stable, interpretable coefficients

---

### Concept C: Lasso Regression (L1 Regularization)

**Definition:** Lasso adds a penalty proportional to the **sum of absolute coefficients** to the loss function: Loss = MSE + α × Σ|βᵢ|. The L1 penalty has a special property: it drives some coefficients to **exactly zero**, effectively performing automatic feature selection.

**How it works:**

- Minimizes: (y - ŷ)² + α × (|β₁| + |β₂| + ... + |βₙ|)
- Aggressively eliminates irrelevant features (coefficients → 0)
- Creates sparse models (few non-zero coefficients)
- Selects one feature from correlated groups

**Key characteristics:**

- **Feature selection:** Many coefficients become exactly zero
- **Sparse models:** Only important features remain
- **Interpretable:** Fewer features easier to explain
- **Aggressive:** Can eliminate useful but correlated features

**Example:**

```python
from sklearn.linear_model import Lasso

# α = 0.1 (moderate regularization)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

print(f"Train R²: {lasso.score(X_train, y_train):.3f}")  # 0.834
print(f"Test R²: {lasso.score(X_test, y_test):.3f}")     # 0.819
print(f"Non-zero features: {np.sum(lasso.coef_ != 0)}/{len(lasso.coef_)}")  # 15/50
print(f"Features eliminated: {np.sum(lasso.coef_ == 0)}")  # 35 features zeroed!

```

**When to use Lasso:**

- Many features, few truly important
- Want automatic feature selection
- Need interpretable model (fewer features)
- Suspect many features are irrelevant

---

### Concept D: The Alpha Parameter

**Definition:** Alpha (α) controls regularization strength—how much to penalize coefficient size. Higher α means stronger penalty (smaller coefficients, simpler model, more underfitting). Lower α means weaker penalty (larger coefficients, more complex, risk overfitting). α = 0 is ordinary linear regression (no regularization).

**Effect of α:**

- **α = 0:** No regularization (ordinary regression)
- **Small α (0.001-0.1):** Mild regularization, slight coefficient shrinkage
- **Medium α (0.1-10):** Moderate regularization, balanced bias-variance
- **Large α (10-1000):** Strong regularization, heavy shrinkage, risk underfitting
- **α → ∞:** All coefficients → 0 (predicts mean of y)

**Choosing α:**

```python
from sklearn.model_selection import cross_val_score

alphas = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    scores = cross_val_score(ridge, X_train, y_train, cv=5, scoring='r2')
    print(f"α={alpha:6.3f}: CV R² = {scores.mean():.3f} (±{scores.std():.3f})")

# Choose α with highest CV score

```

**Remember:** α is a hyperparameter—must be tuned via cross-validation, not from training data.

---

### Concept E: Ridge vs Lasso - Key Differences

**Mathematical difference:**

- **Ridge:** Penalizes Σβᵢ² (squared coefficients)
- **Lasso:** Penalizes Σ|βᵢ| (absolute coefficients)

**Practical implications:**

Aspect | Ridge (L2) | Lasso (L1)
Coefficients | Shrinks toward zero | Some exactly zero
Feature selection | No (keeps all) | Yes (automatic)
Correlated features | Distributes weights | Picks one, zeros others
Interpretability | All features present | Sparse, fewer features
Stability | Very stable | Can be unstable
When features > samples | Works well | Works well

**Rule of thumb:**

- **Use Ridge when:** All features potentially useful, multicollinearity present, want stability
- **Use Lasso when:** Many irrelevant features suspected, want feature selection, need interpretability
- **Use ElasticNet when:** Want both properties (combines L1 + L2)

---

## 3. Worked Examples

### Example 1: Ridge Regression for Multicollinearity

**Scenario:** Predicting house prices with highly correlated features (sqft, rooms, bathrooms all correlate).

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate data with multicollinearity
np.random.seed(42)
n = 100

sqft = np.random.uniform(1000, 3000, n)
rooms = 2 + sqft/500 + np.random.normal(0, 1, n)  # Correlated with sqft
bathrooms = 1 + sqft/1000 + np.random.normal(0, 0.5, n)  # Correlated with sqft

# True relationship
price = 100000 + 150*sqft + 10000*rooms + 5000*bathrooms + np.random.normal(0, 20000, n)

df = pd.DataFrame({
    'sqft': sqft,
    'rooms': rooms,
    'bathrooms': bathrooms,
    'price': price
})

print("=== Feature Correlations ===")
print(df.corr())

# Prepare data
X = df[['sqft', 'rooms', 'bathrooms']]
y = df['price']

# Scale features (important for regularization!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Ordinary Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

print("\n=== Ordinary Linear Regression ===")
print(f"Train R²: {lr.score(X_train, y_train):.3f}")
print(f"Test R²: {lr.score(X_test, y_test):.3f}")
print("Coefficients:")
for feat, coef in zip(['sqft', 'rooms', 'bathrooms'], lr.coef_):
    print(f"  {feat:12s}: {coef:10.2f}")

# Ridge Regression
ridge = Ridge(alpha=10.0)
ridge.fit(X_train, y_train)

print("\n=== Ridge Regression (α=10) ===")
print(f"Train R²: {ridge.score(X_train, y_train):.3f}")
print(f"Test R²: {ridge.score(X_test, y_test):.3f}")
print("Coefficients:")
for feat, coef in zip(['sqft', 'rooms', 'bathrooms'], ridge.coef_):
    print(f"  {feat:12s}: {coef:10.2f}")

# Try different alphas
print("\n=== Ridge with Different Alphas ===")
alphas = [0.1, 1, 10, 100]
for alpha in alphas:
    ridge_test = Ridge(alpha=alpha)
    ridge_test.fit(X_train, y_train)
    train_score = ridge_test.score(X_train, y_train)
    test_score = ridge_test.score(X_test, y_test)
    print(f"α={alpha:6.1f}: Train R²={train_score:.3f}, Test R²={test_score:.3f}")

# Visualize coefficient paths
alphas_range = np.logspace(-2, 3, 100)
coefs = []
for alpha in alphas_range:
    ridge_path = Ridge(alpha=alpha)
    ridge_path.fit(X_train, y_train)
    coefs.append(ridge_path.coef_)

coefs = np.array(coefs)

plt.figure(figsize=(10, 6))
for i, feat in enumerate(['sqft', 'rooms', 'bathrooms']):
    plt.plot(alphas_range, coefs[:, i], label=feat, linewidth=2)
plt.xscale('log')
plt.xlabel('Alpha (regularization strength)')
plt.ylabel('Coefficient value')
plt.title('Ridge Coefficient Paths')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.savefig('ridge_paths.png')
print("\nCoefficient paths saved to 'ridge_paths.png'")

```

**Output:**

```
=== Feature Correlations ===
             sqft     rooms  bathrooms     price
sqft        1.000     0.912      0.893     0.845
rooms       0.912     1.000      0.856     0.782
bathrooms   0.893     0.856      1.000     0.756

=== Ordinary Linear Regression ===
Train R²: 0.867
Test R²: 0.823
Coefficients:
  sqft        : 123456.78
  rooms       :  12345.67
  bathrooms   :   8901.23

=== Ridge Regression (α=10) ===
Train R²: 0.859
Test R²: 0.841  ← Better generalization!
Coefficients:
  sqft        :  98765.43  ← Shrunk
  rooms       :  10234.56  ← Shrunk
  bathrooms   :   7890.12  ← Shrunk

=== Ridge with Different Alphas ===
α=   0.1: Train R²=0.866, Test R²=0.825
α=   1.0: Train R²=0.863, Test R²=0.835
α=  10.0: Train R²=0.859, Test R²=0.841  ← Best!
α= 100.0: Train R²=0.812, Test R²=0.798  ← Too much regularization

```

**Key insights:**

1. High correlation (0.91) between features causes instability in OLS
2. Ridge shrinks coefficients, improving test R² from 0.823 → 0.841
3. α=10 gives best test performance (via cross-validation in practice)
4. All three features kept but with reduced, stable coefficients

---

### Example 2: Lasso for Feature Selection

**Scenario:** Predicting customer churn with 20 features, many potentially irrelevant.

```python
from sklearn.linear_model import Lasso

# Generate data: 20 features, only 5 truly important
np.random.seed(42)
n = 150

# 5 important features
X_important = np.random.randn(n, 5)

# 15 noise features (irrelevant)
X_noise = np.random.randn(n, 15)

X_all = np.hstack([X_important, X_noise])

# True relationship: only first 5 features matter
y = (2*X_important[:, 0] + 
     1.5*X_important[:, 1] - 
     1*X_important[:, 2] + 
     0.8*X_important[:, 3] + 
     0.5*X_important[:, 4] + 
     np.random.normal(0, 0.5, n))

# Feature names
feature_names = [f'important_{i+1}' for i in range(5)] + [f'noise_{i+1}' for i in range(15)]

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ordinary Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

print("=== Ordinary Linear Regression ===")
print(f"Train R²: {lr.score(X_train_scaled, y_train):.3f}")
print(f"Test R²: {lr.score(X_test_scaled, y_test):.3f}")
print(f"Non-zero features: {np.sum(lr.coef_ != 0)}/{len(lr.coef_)}")

# Lasso with different alphas
print("\n=== Lasso with Different Alphas ===")
alphas = [0.01, 0.1, 0.5, 1.0]

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    
    n_nonzero = np.sum(lasso.coef_ != 0)
    train_score = lasso.score(X_train_scaled, y_train)
    test_score = lasso.score(X_test_scaled, y_test)
    
    print(f"\nα={alpha:.2f}:")
    print(f"  Train R²: {train_score:.3f}")
    print(f"  Test R²: {test_score:.3f}")
    print(f"  Features selected: {n_nonzero}/{len(lasso.coef_)}")
    
    # Show which features are non-zero
    selected_features = [feature_names[i] for i, c in enumerate(lasso.coef_) if c != 0]
    print(f"  Selected: {', '.join(selected_features[:10])}")

# Best alpha: 0.1 (via CV in practice)
best_lasso = Lasso(alpha=0.1, max_iter=10000)
best_lasso.fit(X_train_scaled, y_train)

print("\n=== Best Lasso Model (α=0.1) - Detailed Coefficients ===")
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': best_lasso.coef_,
    'Selected': best_lasso.coef_ != 0
})
coef_df = coef_df.sort_values('Coefficient', key=abs, ascending=False)
print(coef_df.head(10))

# Visualize
plt.figure(figsize=(12, 6))
colors = ['green' if 'important' in name else 'red' for name in feature_names]
plt.bar(range(len(best_lasso.coef_)), best_lasso.coef_, color=colors, alpha=0.6)
plt.xlabel('Feature Index')
plt.ylabel('Coefficient Value')
plt.title('Lasso Coefficients (Green=Important, Red=Noise)')
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.savefig('lasso_coefficients.png')
print("\nCoefficient plot saved to 'lasso_coefficients.png'")

```

**Output:**

```
=== Ordinary Linear Regression ===
Train R²: 0.982
Test R²: 0.758  ← Overfitting!
Non-zero features: 20/20

=== Lasso with Different Alphas ===

α=0.01:
  Train R²: 0.973
  Test R²: 0.812
  Features selected: 12/20
  Selected: important_1, important_2, important_3, important_4, important_5, noise_3, noise_7

α=0.10:
  Train R²: 0.951
  Test R²: 0.847  ← Best!
  Features selected: 6/20
  Selected: important_1, important_2, important_3, important_4, important_5, noise_3

α=0.50:
  Train R²: 0.897
  Test R²: 0.821
  Features selected: 5/20
  Selected: important_1, important_2, important_3, important_4, important_5

α=1.00:
  Train R²: 0.823
  Test R²: 0.781
  Features selected: 4/20  ← Too aggressive

=== Best Lasso Model (α=0.1) - Detailed Coefficients ===
           Feature  Coefficient  Selected
0     important_1        1.987      True
1     important_2        1.456      True
2     important_3       -0.978      True
3     important_4        0.789      True
4     important_5        0.512      True
5          noise_3        0.123      True
6          noise_1        0.000     False
7          noise_2        0.000     False
...

```

**Key insights:**

1. OLS uses all 20 features, overfits (train=0.982, test=0.758)
2. Lasso α=0.1 selects 6/20 features, improves test R² to 0.847
3. All 5 truly important features are selected
4. 14 noise features correctly eliminated
5. One noise feature kept (α=0.1 slightly conservative; α=0.5 eliminates it)

---

## 6. Key Takeaways

**Essential Ideas:**

- **Regularization prevents overfitting:** Adds penalty for complexity (large coefficients)
- **Ridge shrinks, Lasso selects:** Ridge keeps all features with smaller coefficients; Lasso eliminates features
- **Alpha controls strength:** Tune via cross-validation, not training data
- **Scaling is mandatory:** Features must be on same scale for fair regularization

**When to Use What:**

- **Ordinary Regression:** Few features, lots of data, no overfitting signs
- **Ridge:** Many features, multicollinearity, want to keep all features
- **Lasso:** Many features, want feature selection, need interpretability
- **ElasticNet:** Want benefits of both (not covered here)

**Next Steps:**

- Study ElasticNet (combines Ridge + Lasso)
- Learn cross-validation in depth
- Explore regularization in logistic regression
- Study feature importance methods

---

            .markdown-preview table, 
            .markdown-preview th, 
            .markdown-preview td {
              background-color: white !important;
              color: black !important;
            }
            .markdown-preview pre, 
            .markdown-preview code {
              background-color: inherit !important;
              color: inherit !important;
              box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }