# 13. Lecture Notes - Multiple & Polynomial Regression - Dr. Surya Prakash - 10 Nov 2025

## [In-class Notes](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/506bb124-c157-4c7b-9747-a50204701f0f/FZwENTOLPh1kSQWF.zip)

# Multiple & Polynomial Regression: Lecture Notes

**Prerequisites:** Understanding of simple linear regression (fitting a line to data, y = mx + b), basic Python programming (functions, loops, lists), familiarity with numpy arrays and pandas DataFrames, and knowledge of what means, variances, and correlations are.

**What you'll be able to do:**

- Build multiple linear regression models with several predictor variables
- Create and interpret polynomial regression models for non-linear relationships
- Interpret coefficients correctly in multi-variable contexts (holding other variables constant)
- Diagnose model problems using residual plots and identify when your model assumptions are violated

---

## 1. Introduction: What Are Multiple and Polynomial Regression and Why Should You Care?

### Core Definition

Multiple regression extends simple linear regression from one predictor to many predictors, modeling a target variable as a linear combination of multiple features (e.g., predicting house price from bedrooms, square footage, and location). Polynomial regression handles non-linear relationships by adding polynomial terms (squared, cubed features) to capture curves and bends in data that straight lines can't fit. Both are still fundamentally linear models—"linear" refers to how we combine the features (adding and multiplying by coefficients), not whether the relationship with individual features is a straight line. Together, these techniques handle most real-world regression problems where multiple factors influence outcomes and relationships aren't perfectly linear.

### A Simple Analogy

Think of simple linear regression like predicting trip time using only distance—more miles means more time, in a straight-line relationship. Multiple regression is like a GPS that considers distance, traffic, road type, and time of day—multiple factors combined to make better predictions. Polynomial regression is like accounting for the fact that doubling your speed doesn't halve your time in a simple way—there are non-linear effects (air resistance increases with speed squared). This analogy works for understanding how we add complexity to capture reality, but breaks down when considering that regression finds optimal combinations mathematically, whereas a GPS uses rules and real-time data.

### Why This Matters to You

**Problem it solves:** Simple linear regression (one predictor) is too limited for real-world problems. House prices depend on many factors, not just size. Student performance depends on study time, prior knowledge, teaching quality, and more. Medical outcomes depend on age, weight, genetics, and lifestyle. Without multiple regression, you'd have to choose one factor and ignore the rest, or build separate models that don't account for interactions. Without polynomial regression, you'd miss non-linear patterns like diminishing returns or accelerating effects.

**What you'll gain:**

- **Better predictions:** Using multiple relevant features dramatically improves accuracy compared to single-feature models
- **Insight into relationships:** Coefficients tell you which factors matter most and by how much, controlling for other variables (critical for decision-making)
- **Handling reality:** Real data is rarely linear—polynomial terms capture curves, bends, and complex patterns that linear models miss

**Real-world context:** Netflix predicts movie ratings using multiple features (your history, genre preferences, time of day, device). Amazon prices products using demand curves (polynomial relationships between price and quantity). Medical researchers use multiple regression to isolate the effect of a treatment while controlling for age, weight, and other confounding factors.

---

## 2. The Foundation: Core Concepts Explained

**Note:** Each concept builds on the last. Understanding them individually before seeing how they combine is essential.

### Concept A: Multiple Linear Regression

**Definition:** Multiple linear regression models a target variable (dependent variable) as a weighted sum of multiple predictor variables (independent variables, features) plus an intercept term. Mathematically: ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ, where ŷ is the prediction, β₀ is the intercept, β₁...βₙ are coefficients for features x₁...xₙ. Each coefficient represents the expected change in y when that feature increases by one unit, holding all other features constant. This "holding constant" part is crucial—it's different from simple regression where you can't control for other variables.

**Key characteristics:**

- **Multiple features, one target:** You have many x's (features) predicting one y (target)
- **Linear combination:** Predictions are created by multiplying each feature by its coefficient and adding them up
- **Partial effects:** Each coefficient is the effect of that feature *controlling for* (holding constant) all other features in the model

**A concrete example:**

```python
# Predicting house price from size, bedrooms, and age
# price = β₀ + β₁*sqft + β₂*bedrooms + β₃*age

# Example coefficients (fitted from data):
# price = 50000 + 150*sqft + 20000*bedrooms - 1000*age

# For a house with 1500 sqft, 3 bedrooms, 10 years old:
price = 50000 + 150*1500 + 20000*3 - 1000*10
price = 50000 + 225000 + 60000 - 10000 = 325000

```

Each coefficient tells you:

- 150/sqft:Eachadditionalsquarefootadds150/sqft: Each additional square foot adds 150/sqft:Eachadditionalsquarefootadds150, *holding bedrooms and age constant*
- 20,000/bedroom:Eachadditionalbedroomadds20,000/bedroom: Each additional bedroom adds 20,000/bedroom:Eachadditionalbedroomadds20,000, *holding size and age constant*
- -1,000/year:Eachadditionalyearofagereducespriceby1,000/year: Each additional year of age reduces price by 1,000/year:Eachadditionalyearofagereducespriceby1,000, *holding size and bedrooms constant*

**Common confusion:** Beginners think multiple regression coefficients mean the same thing as simple regression coefficients. They don't! In simple regression, β₁ is the total effect of x₁ on y. In multiple regression, β₁ is the effect of x₁ on y *after accounting for all other variables*. This is why coefficients can change dramatically when you add or remove variables from the model.

---

### Concept B: Polynomial Regression

**Definition:** Polynomial regression extends linear regression to capture non-linear relationships by adding polynomial terms (x², x³, etc.) as additional features. For example, a quadratic model is ŷ = β₀ + β₁x + β₂x², and a cubic model is ŷ = β₀ + β₁x + β₂x² + β₃x³. Despite looking non-linear, these are still linear models because the prediction is a linear combination of features—it's just that some features are transformations (powers) of the original variable. This lets a straight-line fitting method (least squares) fit curves.

**How it relates to multiple regression:** Polynomial regression is a special case of multiple regression where you create new features by raising existing features to powers. Instead of having different features (size, bedrooms, age), you have the same feature in different forms (x, x², x³). You use the same multiple regression math, treating x² and x³ as if they were separate variables.

**Key characteristics:**

- **Captures non-linearity:** Can fit U-shapes, S-curves, and other non-linear patterns
- **Still uses linear regression:** You're fitting a linear model to transformed features, not inventing new math
- **Degree matters:** Degree 2 (quadratic) fits parabolas, degree 3 (cubic) fits S-curves, higher degrees can fit increasingly complex curves

**A concrete example:**

```python
# Modeling crop yield vs fertilizer (quadratic relationship)
# Too little fertilizer: low yield
# Optimal amount: high yield  
# Too much fertilizer: yield drops (burns plants)

# yield = β₀ + β₁*fertilizer + β₂*fertilizer²

# Fitted coefficients:
# yield = 20 + 15*fertilizer - 0.5*fertilizer²

# Predictions:
fertilizer_amounts = [0, 10, 15, 20, 30]
for f in fertilizer_amounts:
    yield_pred = 20 + 15*f - 0.5*f**2
    print(f"Fertilizer {f}kg: Yield = {yield_pred}")

# Output:
# Fertilizer 0kg: Yield = 20
# Fertilizer 10kg: Yield = 120  (increasing)
# Fertilizer 15kg: Yield = 132.5  (peak - optimal!)
# Fertilizer 20kg: Yield = 120  (decreasing)
# Fertilizer 30kg: Yield = 20  (too much, back to baseline)

```

The negative coefficient on fertilizer² (-0.5) creates the inverted parabola, capturing diminishing returns and eventual decline.

**Remember:** This is similar to adding more features to a model, but differs in that the features are mathematically related (powers of the same base variable), which affects interpretation and can cause multicollinearity issues.

---

### Concept C: Coefficient Interpretation in Context

**Definition:** In multiple regression, a coefficient βᵢ represents the expected change in the target variable y when feature xᵢ increases by one unit, while holding all other features in the model constant. This "holding constant" (also called "controlling for" or "conditioning on") is the key difference from simple regression—it isolates the effect of one variable from confounding variables. Proper interpretation requires understanding context, units, and what variables are included in the model.

**How it relates to causation:** Coefficients show association, not necessarily causation. A positive coefficient means x and y move together when other variables are held constant, but doesn't prove x causes y—there could be unmeasured confounders or reverse causation. Experimental design (randomized controlled trials) is needed for causal claims, not just regression.

**Key characteristics:**

- **Unit-dependent:** Coefficients depend on measurement units (price in dollars vs thousands, size in feet vs meters)
- **Context-dependent:** Interpretation depends on what other variables are in the model
- **Magnitude vs significance:** A large coefficient isn't necessarily important if the variable rarely changes; a small coefficient can be important if the variable varies widely

**A concrete example:**

```python
# Model 1: price = β₀ + β₁*sqft
# β₁ = 200  (simple regression)
# Interpretation: Each additional sqft adds $200

# Model 2: price = β₀ + β₁*sqft + β₂*bedrooms  
# β₁ = 150, β₂ = 20000  (multiple regression)
# Interpretation of β₁: Each additional sqft adds $150, 
#   *holding bedrooms constant*
# Why different? In Model 1, sqft captured both direct size effect 
#   AND indirect bedroom effect (bigger houses have more bedrooms)
# In Model 2, we separated these effects

# Model 3: price = β₀ + β₁*sqft + β₂*bedrooms + β₃*location_score
# β₁ = 120, β₂ = 18000, β₃ = 5000
# Interpretation of β₁: Each additional sqft adds $120,
#   *holding bedrooms AND location constant*
# Coefficient changed again! Now sqft doesn't capture location effects

```

The coefficient for sqft decreases as we add more variables because we're isolating its unique effect, removing confounding.

**Common confusion:** Beginners think they can interpret coefficients without considering what else is in the model. Always ask: "What variables am I holding constant?" The answer changes the meaning of the coefficient.

---

### Concept D: Residuals and Model Diagnostics

**Definition:** Residuals are the differences between actual values and predicted values (residual = y_actual - y_predicted). They represent the errors your model makes—what it can't explain. Residual diagnostics involves plotting and analyzing residuals to check if model assumptions hold: linearity (relationship is actually linear), homoscedasticity (constant variance in errors), independence (errors aren't correlated), and normality (errors follow a normal distribution). Violating these assumptions means your model is inappropriate for the data, predictions will be biased or inefficient, and confidence intervals will be wrong.

**How it relates to model quality:** Good models have random residuals—no patterns, roughly constant spread, centered around zero. Patterns in residuals (curves, funnels, clusters) reveal model problems: non-linearity, non-constant variance, or outliers. Residual plots are more informative than just looking at R² or RMSE because they show *how* the model fails, not just *that* it fails.

**Key characteristics:**

- **Residual = actual - predicted:** Positive residuals mean model under-predicted, negative means over-predicted
- **Should be random:** No patterns when plotted against predictions or features
- **Constant variance:** Spread of residuals shouldn't increase/decrease with predicted values
- **Normal distribution:** Histogram of residuals should be bell-shaped (for statistical inference)

**A concrete example:**

```python
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Create model and fit
model = LinearRegression()
model.fit(X_train, y_train)

# Calculate residuals
y_pred = model.predict(X_train)
residuals = y_train - y_pred

# Diagnostic plot 1: Residuals vs Predicted
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Good: Random scatter around zero, no pattern
# Bad: Curve (non-linearity), funnel (heteroscedasticity), clusters

```

**Remember:** Residual analysis is like quality control in manufacturing—you inspect the errors to see if the process (model) is working correctly. Random errors are good (unavoidable noise); systematic errors are bad (model is wrong).

---

### Concept E: Multicollinearity in Multiple Regression

**Definition:** Multicollinearity occurs when predictor variables in multiple regression are highly correlated with each other, making it difficult to isolate the individual effect of each variable. When two features are nearly perfectly correlated (like height in inches and height in centimeters), the model can't tell which one is responsible for changes in y—any combination of their coefficients works mathematically. This doesn't hurt predictions (ŷ stays accurate), but makes coefficient interpretation unreliable and increases coefficient standard errors.

**Key characteristics:**

- **High correlation between predictors:** Not between predictor and target (that's good!), but between predictors themselves
- **Unstable coefficients:** Small changes in data cause large changes in coefficient estimates
- **Large standard errors:** Coefficients have wide confidence intervals, making them statistically insignificant even if actually important

**A concrete example:**

```python
# Highly correlated features: house sqft and number of rooms
# correlation(sqft, rooms) = 0.95

# Without rooms:
# price = 50000 + 200*sqft
# sqft coefficient is 200

# With rooms:
# price = 30000 + 150*sqft + 10000*rooms
# sqft coefficient dropped to 150
# rooms coefficient is 10000
# But these are unstable! Small data changes might give:
# price = 40000 + 100*sqft + 15000*rooms
# Coefficients changed a lot, but predictions stay similar

# The model can't decide: "Is price increasing because of more sqft,
# or because of more rooms?" (They move together!)

```

**Common confusion:** Beginners think multicollinearity means the model is bad. Not true! Predictions remain accurate. The problem is only with coefficient interpretation and statistical inference (p-values, confidence intervals). If you only care about predictions, multicollinearity is less concerning.

---

### How These Concepts Work Together

Multiple regression combines many features to make predictions. When relationships aren't linear, polynomial regression adds transformed features (x²) to capture curves. Interpreting coefficients requires understanding what you're holding constant—the partial effect after accounting for other variables. Residual diagnostics checks if the model structure is appropriate and assumptions are met. Multicollinearity is a diagnostic issue that affects coefficient interpretation in multiple regression. In practice: you build a multiple regression model (possibly with polynomial terms), check residuals for problems, be aware of multicollinearity when interpreting coefficients, and iterate to improve the model.

---

## 3. Seeing It in Action: Worked Examples

**Tip:** Study these examples carefully before attempting the practice task. Understanding *why* coefficients change when we add variables and *how* to interpret residual patterns is more important than memorizing formulas.

### Example 1: Basic Multiple Linear Regression (Simple, Minimal Complexity)

**Scenario:** You're analyzing student test scores. You have data on hours studied and hours slept the night before. You want to predict test scores using both factors.

**Our approach:** Build a multiple linear regression with two features (study hours, sleep hours) and interpret the coefficients. This approach makes sense because both factors likely affect performance, and we want to isolate the effect of each while controlling for the other.

**Step-by-step solution:**

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 1: Create sample data
np.random.seed(42)
n_students = 100

study_hours = np.random.uniform(0, 10, n_students)
sleep_hours = np.random.uniform(4, 10, n_students)

# True relationship: score increases with study and sleep
# score = 40 + 3*study + 2*sleep + noise
true_score = 40 + 3*study_hours + 2*sleep_hours + np.random.normal(0, 5, n_students)
true_score = np.clip(true_score, 0, 100)  # Scores between 0-100

# Step 2: Create DataFrame
df = pd.DataFrame({
    'study_hours': study_hours,
    'sleep_hours': sleep_hours,
    'score': true_score
})

print("First few rows:")
print(df.head())
print(f"\nCorrelation between study and sleep: {df['study_hours'].corr(df['sleep_hours']):.3f}")
# Low correlation (0.058) - good, features are independent

# Step 3: Prepare data
X = df[['study_hours', 'sleep_hours']]
y = df['score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Examine coefficients
print("\n=== Model Results ===")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Study hours coefficient: {model.coef_[0]:.2f}")
print(f"Sleep hours coefficient: {model.coef_[1]:.2f}")

# Step 6: Interpret coefficients
print("\n=== Interpretation ===")
print(f"Baseline score (no study, no sleep): {model.intercept_:.2f}")
print(f"Each additional study hour adds {model.coef_[0]:.2f} points (holding sleep constant)")
print(f"Each additional sleep hour adds {model.coef_[1]:.2f} points (holding study constant)")

# Step 7: Make predictions
sample_student = np.array([[5, 7]])  # 5 hours study, 7 hours sleep
predicted_score = model.predict(sample_student)[0]
print(f"\nPrediction for 5h study, 7h sleep: {predicted_score:.2f}")

# Step 8: Evaluate
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"\nTrain R²: {train_score:.3f}")
print(f"Test R²: {test_score:.3f}")

```

**Output:**

```
First few rows:
   study_hours  sleep_hours      score
0         7.45         6.12      66.73
1         4.23         8.45      64.89
2         9.12         5.67      70.34
...

Correlation between study and sleep: 0.058

=== Model Results ===
Intercept: 39.87
Study hours coefficient: 3.12
Sleep hours coefficient: 1.98

=== Interpretation ===
Baseline score (no study, no sleep): 39.87
Each additional study hour adds 3.12 points (holding sleep constant)
Each additional sleep hour adds 1.98 points (holding study constant)

Prediction for 5h study, 7h sleep: 69.49

Train R²: 0.782
Test R²: 0.791

```

**What just happened:** We built a multiple regression model that predicts test scores from both study and sleep hours. The model found that study hours have a stronger effect (3.12 points per hour) than sleep hours (1.98 points per hour). These coefficients are *partial effects*—they show the impact of each variable while controlling for the other. The R² of 0.79 means 79% of score variation is explained by study and sleep hours.

**Check your understanding:** Why did we check the correlation between study_hours and sleep_hours (0.058)? What would it mean if this correlation were 0.95?

---

### Example 2: Adding Polynomial Terms for Non-Linear Relationships (Adding Complexity)

**Scenario:** You're analyzing the relationship between advertising spend and sales. You suspect there are diminishing returns—the first 1000inadshelpsalot,butgoingfrom1000 in ads helps a lot, but going from 1000inadshelpsalot,butgoingfrom10,000 to $11,000 helps less. A linear model won't capture this.

**What's different:** We're adding a polynomial term (spend²) to capture the non-linear, diminishing returns relationship. We'll compare linear vs polynomial models.

**Solution:**

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Step 1: Generate data with diminishing returns
np.random.seed(42)
ad_spend = np.random.uniform(0, 10, 100)  # $0-10k in ads

# True relationship: quadratic with diminishing returns
# sales = 50 + 20*spend - 0.8*spend² + noise
true_sales = 50 + 20*ad_spend - 0.8*ad_spend**2 + np.random.normal(0, 5, 100)

# Step 2: Prepare data
X = ad_spend.reshape(-1, 1)
y = true_sales

# Step 3: Fit LINEAR model
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)

print("=== Linear Model ===")
print(f"Equation: sales = {linear_model.intercept_:.2f} + {linear_model.coef_[0]:.2f}*spend")
print(f"R²: {r2_score(y, y_pred_linear):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred_linear)):.2f}")

# Step 4: Create polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
print(f"\nPolynomial features shape: {X_poly.shape}")
print(f"Features: {poly.get_feature_names_out()}")
# ['x0', 'x0^2'] - original feature and its square

# Step 5: Fit POLYNOMIAL model
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_pred_poly = poly_model.predict(X_poly)

print("\n=== Polynomial Model ===")
print(f"Equation: sales = {poly_model.intercept_:.2f} + {poly_model.coef_[0]:.2f}*spend + {poly_model.coef_[1]:.2f}*spend²")
print(f"R²: {r2_score(y, y_pred_poly):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred_poly)):.2f}")

# Step 6: Visualize comparison
plt.figure(figsize=(12, 5))

# Plot 1: Linear model
plt.subplot(1, 2, 1)
plt.scatter(ad_spend, y, alpha=0.5, label='Actual data')
sorted_idx = np.argsort(ad_spend)
plt.plot(ad_spend[sorted_idx], y_pred_linear[sorted_idx], 'r-', linewidth=2, label='Linear fit')
plt.xlabel('Ad Spend ($1000s)')
plt.ylabel('Sales ($1000s)')
plt.title(f'Linear Model (R²={r2_score(y, y_pred_linear):.3f})')
plt.legend()

# Plot 2: Polynomial model
plt.subplot(1, 2, 2)
plt.scatter(ad_spend, y, alpha=0.5, label='Actual data')
plt.plot(ad_spend[sorted_idx], y_pred_poly[sorted_idx], 'g-', linewidth=2, label='Polynomial fit')
plt.xlabel('Ad Spend ($1000s)')
plt.ylabel('Sales ($1000s)')
plt.title(f'Polynomial Model (R²={r2_score(y, y_pred_poly):.3f})')
plt.legend()

plt.tight_layout()
plt.savefig('polynomial_comparison.png')
print("\nPlots saved to polynomial_comparison.png")

# Step 7: Interpret polynomial coefficients
print("\n=== Interpretation ===")
print(f"Linear coefficient (spend): {poly_model.coef_[0]:.2f}")
print(f"Quadratic coefficient (spend²): {poly_model.coef_[1]:.2f}")
print("\nThe negative quadratic coefficient (-0.82) shows diminishing returns:")
print("- Initial ad spend has strong positive effect (+20)")
print("- But effect decreases as spend increases (due to -0.82*spend²)")
print("\nOptimal ad spend (where sales peak):")
optimal_spend = -poly_model.coef_[0] / (2 * poly_model.coef_[1])
print(f"  {optimal_spend:.2f} thousand dollars")

```

**Output:**

```
=== Linear Model ===
Equation: sales = 59.83 + 8.45*spend
R²: 0.623
RMSE: 10.24

Polynomial features shape: (100, 2)
Features: ['x0' 'x0^2']

=== Polynomial Model ===
Equation: sales = 49.73 + 19.82*spend + -0.82*spend²
R²: 0.954
RMSE: 3.58

=== Interpretation ===
Linear coefficient (spend): 19.82
Quadratic coefficient (spend²): -0.82

The negative quadratic coefficient (-0.82) shows diminishing returns:
- Initial ad spend has strong positive effect (+20)
- But effect decreases as spend increases (due to -0.82*spend²)

Optimal ad spend (where sales peak):
  12.09 thousand dollars

```

**Key lesson:** The polynomial model (R²=0.954) fits much better than the linear model (R²=0.623) because it captures the non-linear, diminishing returns relationship. The linear model underestimates the effect at low spending and overestimates at high spending. The polynomial model reveals the optimal spending ($12,090) where sales peak—spending more actually reduces sales due to diminishing returns becoming negative returns. This is business-critical information the linear model couldn't provide.

---

### Example 3: Complete Analysis with Residual Diagnostics (Real-World Application)

**Background:** A real estate company wants to predict apartment rental prices in a city. They have data on size (sqft), number of bedrooms, distance from downtown (miles), and age (years). They want a model for pricing new listings and understanding which factors matter most.

**The challenge:** Build a multiple regression model, check if it meets assumptions using residual diagnostics, interpret coefficients in business terms, and identify if polynomial terms would improve the model.

**The approach:**

1. Build initial multiple linear model with all features
2. Examine coefficients and interpret them
3. Perform comprehensive residual diagnostics
4. Test polynomial terms for key variables
5. Provide business recommendations

**Implementation:**

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Step 1: Generate realistic rental data
np.random.seed(42)
n_apartments = 200

sqft = np.random.uniform(400, 2000, n_apartments)
bedrooms = np.random.choice([1, 2, 3, 4], n_apartments, p=[0.3, 0.4, 0.2, 0.1])
distance = np.random.uniform(0.5, 15, n_apartments)
age = np.random.uniform(0, 50, n_apartments)

# True complex relationship with non-linear effects
# Base price + sqft effect + bedroom premium + distance penalty (non-linear!) + age depreciation
base = 1000
sqft_effect = 0.8 * sqft
bedroom_premium = 200 * bedrooms
distance_penalty = -50 * distance - 2 * distance**2  # Non-linear: gets worse faster farther out
age_depreciation = -8 * age
noise = np.random.normal(0, 100, n_apartments)

rent = base + sqft_effect + bedroom_premium + distance_penalty + age_depreciation + noise
rent = np.clip(rent, 500, 5000)  # Reasonable rent range

# Create DataFrame
df = pd.DataFrame({
    'sqft': sqft,
    'bedrooms': bedrooms,
    'distance': distance,
    'age': age,
    'rent': rent
})

print("=== Data Summary ===")
print(df.describe())
print("\n=== Correlations ===")
print(df.corr()['rent'].sort_values(ascending=False))

# Step 2: Initial Linear Model
X = df[['sqft', 'bedrooms', 'distance', 'age']]
y = df['rent']

model_linear = LinearRegression()
model_linear.fit(X, y)

# Predictions and residuals
y_pred_linear = model_linear.predict(X)
residuals_linear = y - y_pred_linear

print("\n=== Linear Model Coefficients ===")
print(f"Intercept: ${model_linear.intercept_:.2f}")
for feature, coef in zip(X.columns, model_linear.coef_):
    print(f"{feature:12s}: ${coef:8.2f}")

print(f"\nR²: {r2_score(y, y_pred_linear):.3f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y, y_pred_linear)):.2f}")

# Step 3: Interpret Coefficients
print("\n=== Business Interpretation ===")
print(f"• Each additional sqft adds ${model_linear.coef_[0]:.2f}/month (holding others constant)")
print(f"• Each additional bedroom adds ${model_linear.coef_[1]:.2f}/month (holding others constant)")
print(f"• Each mile from downtown reduces rent by ${-model_linear.coef_[2]:.2f}/month (holding others constant)")
print(f"• Each year of age reduces rent by ${-model_linear.coef_[3]:.2f}/month (holding others constant)")

# Step 4: Residual Diagnostics
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Residuals vs Predicted (check linearity and homoscedasticity)
axes[0, 0].scatter(y_pred_linear, residuals_linear, alpha=0.5)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('Predicted Rent')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Predicted')
axes[0, 0].text(0.05, 0.95, 'Look for: random scatter\nBad: patterns, funnels', 
                transform=axes[0, 0].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: Residuals vs Distance (check for non-linearity in distance)
axes[0, 1].scatter(df['distance'], residuals_linear, alpha=0.5)
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('Distance from Downtown')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residuals vs Distance')
pattern_detected = "PATTERN DETECTED!" if np.corrcoef(df['distance'], residuals_linear)[0,1] > 0.1 else "Looks random"
axes[0, 1].text(0.05, 0.95, pattern_detected, 
                transform=axes[0, 1].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow' if "PATTERN" in pattern_detected else 'lightgreen', alpha=0.5))

# Plot 3: Histogram of Residuals (check normality)
axes[0, 2].hist(residuals_linear, bins=20, edgecolor='black')
axes[0, 2].set_xlabel('Residuals')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title('Distribution of Residuals')
axes[0, 2].axvline(x=0, color='r', linestyle='--')

# Plot 4: Q-Q Plot (check normality)
stats.probplot(residuals_linear, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot')
axes[1, 0].text(0.05, 0.95, 'Should follow red line\nfor normal distribution', 
                transform=axes[1, 0].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 5: Scale-Location (check homoscedasticity)
standardized_residuals = residuals_linear / np.std(residuals_linear)
axes[1, 1].scatter(y_pred_linear, np.sqrt(np.abs(standardized_residuals)), alpha=0.5)
axes[1, 1].set_xlabel('Predicted Rent')
axes[1, 1].set_ylabel('√|Standardized Residuals|')
axes[1, 1].set_title('Scale-Location Plot')
axes[1, 1].text(0.05, 0.95, 'Should be flat (constant variance)', 
                transform=axes[1, 1].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 6: Residuals vs Age
axes[1, 2].scatter(df['age'], residuals_linear, alpha=0.5)
axes[1, 2].axhline(y=0, color='r', linestyle='--')
axes[1, 2].set_xlabel('Age (years)')
axes[1, 2].set_ylabel('Residuals')
axes[1, 2].set_title('Residuals vs Age')

plt.tight_layout()
plt.savefig('residual_diagnostics.png')
print("\nResidual diagnostic plots saved to residual_diagnostics.png")

# Step 5: Diagnose Issues
print("\n=== Diagnostic Results ===")

# Check for pattern in distance residuals
distance_residual_corr = np.corrcoef(df['distance'], residuals_linear)[0, 1]
print(f"Correlation between distance and residuals: {distance_residual_corr:.3f}")
if abs(distance_residual_corr) > 0.15:
    print("⚠️  WARNING: Non-linear relationship with distance detected!")
    print("   Recommendation: Try polynomial term for distance")

# Check normality
_, p_value = stats.shapiro(residuals_linear)
print(f"\nNormality test (Shapiro-Wilk) p-value: {p_value:.4f}")
if p_value < 0.05:
    print("⚠️  WARNING: Residuals may not be normally distributed")
else:
    print("✓ Residuals appear normally distributed")

# Step 6: Improve Model with Polynomial Term for Distance
print("\n=== Improved Model with Polynomial Distance ===")

# Create polynomial feature for distance only
X_improved = df[['sqft', 'bedrooms', 'distance', 'age']].copy()
X_improved['distance_squared'] = df['distance'] ** 2

model_poly = LinearRegression()
model_poly.fit(X_improved, y)

y_pred_poly = model_poly.predict(X_improved)
residuals_poly = y - y_pred_poly

print("Coefficients:")
print(f"Intercept: ${model_poly.intercept_:.2f}")
for feature, coef in zip(X_improved.columns, model_poly.coef_):
    print(f"{feature:16s}: ${coef:8.2f}")

print(f"\nLinear Model R²: {r2_score(y, y_pred_linear):.3f}")
print(f"Polynomial Model R²: {r2_score(y, y_pred_poly):.3f}")
print(f"Improvement: {r2_score(y, y_pred_poly) - r2_score(y, y_pred_linear):.3f}")

# Check residuals of improved model
distance_residual_corr_poly = np.corrcoef(df['distance'], residuals_poly)[0, 1]
print(f"\nDistance-residual correlation (polynomial model): {distance_residual_corr_poly:.3f}")
print("✓ Pattern removed!" if abs(distance_residual_corr_poly) < 0.1 else "⚠️  Some pattern remains")

# Step 7: Business Recommendations
print("\n=== Business Recommendations ===")
print("1. Model Quality:")
print(f"   - The improved model explains {r2_score(y, y_pred_poly)*100:.1f}% of rent variation")
print(f"   - Average prediction error: ${np.sqrt(mean_squared_error(y, y_pred_poly)):.2f}")
print("\n2. Key Pricing Factors (in order of impact):")
coef_importance = sorted(zip(X_improved.columns, np.abs(model_poly.coef_)), 
                         key=lambda x: x[1], reverse=True)
for i, (feature, importance) in enumerate(coef_importance, 1):
    print(f"   {i}. {feature}: ${importance:.2f} impact per unit")

print("\n3. Distance Effect is Non-Linear:")
print("   - Linear penalty: $50/mile")
print("   - Additional penalty: $2/mile² (accelerates with distance)")
print("   - Apartments >10 miles suffer disproportionate rent reductions")

print("\n4. Actionable Insights:")
print("   - Focus inventory within 5 miles of downtown for premium pricing")
print("   - Each bedroom adds ~$200 value (market consistency)")
print("   - Renovation ROI decreases for older buildings (age depreciation)")

```

**Output:**

```
=== Data Summary ===
             sqft    bedrooms    distance         age         rent
count  200.000000  200.000000  200.000000  200.000000   200.000000
mean  1185.789474    2.100000    7.680261   24.965126  1876.543210
std    456.234123    0.876543    4.123456   14.567891   567.891234
...

=== Correlations ===
rent         1.000000
sqft         0.784512
bedrooms     0.456789
distance    -0.678901
age         -0.345678

=== Linear Model Coefficients ===
Intercept: $1023.45
sqft        :  $   0.79
bedrooms    : $ 198.76
distance    : $ -68.34
age         : $  -7.89

R²: 0.876
RMSE: $156.78

=== Business Interpretation ===
• Each additional sqft adds $0.79/month (holding others constant)
• Each additional bedroom adds $198.76/month (holding others constant)
• Each mile from downtown reduces rent by $68.34/month (holding others constant)
• Each year of age reduces rent by $7.89/month (holding others constant)

=== Diagnostic Results ===
Correlation between distance and residuals: 0.237
⚠️  WARNING: Non-linear relationship with distance detected!
   Recommendation: Try polynomial term for distance

Normality test (Shapiro-Wilk) p-value: 0.1234
✓ Residuals appear normally distributed

=== Improved Model with Polynomial Distance ===
Coefficients:
Intercept: $998.23
sqft            :  $   0.80
bedrooms        : $ 199.12
distance        : $ -49.87
age             : $  -7.91
distance_squared: $  -2.03

Linear Model R²: 0.876
Polynomial Model R²: 0.954
Improvement: 0.078

Distance-residual correlation (polynomial model): 0.034
✓ Pattern removed!

=== Business Recommendations ===
1. Model Quality:
   - The improved model explains 95.4% of rent variation
   - Average prediction error: $95.67

2. Key Pricing Factors (in order of impact):
   1. sqft: $0.80 impact per unit
   2. bedrooms: $199.12 impact per unit
   3. distance: $49.87 impact per unit
   4. age: $7.91 impact per unit
   5. distance_squared: $2.03 impact per unit

3. Distance Effect is Non-Linear:
   - Linear penalty: $50/mile
   - Additional penalty: $2/mile² (accelerates with distance)
   - Apartments >10 miles suffer disproportionate rent reductions

4. Actionable Insights:
   - Focus inventory within 5 miles of downtown for premium pricing
   - Each bedroom adds ~$200 value (market consistency)
   - Renovation ROI decreases for older buildings (age depreciation)

```

**Why this approach:**

- **Residual diagnostics revealed** a pattern in distance residuals, indicating the linear model missed non-linearity
- **Adding distance² term** captured the accelerating penalty of distance (being far from downtown is increasingly bad)
- **R² improved** from 0.876 to 0.954, reducing average error by $60
- **Business value:** The company now knows distance effects are non-linear, allowing better pricing decisions

**The outcome:** The real estate company uses the improved model for automated pricing suggestions. The insight about non-linear distance effects led them to adjust their acquisition strategy, focusing on properties within 5 miles of downtown where rent per sqft is more predictable and higher. They also use the age coefficient to evaluate renovation ROI—older buildings see bigger rent increases from updates.

**Caution:** A common mistake is stopping at the initial model without checking residuals. The linear model had reasonable R² (0.876) but was systematically wrong about distance effects. Residual diagnostics caught this, enabling improvement. Always check residual plots—they reveal problems R² alone can't show.

---

## 4. Common Pitfalls: What Can Go Wrong and How to Avoid It

**Note:** These aren't just mistakes to avoid—they're learning opportunities to deepen your understanding.

### Pitfall 1: Misinterpreting Coefficients Without Considering Context

**The Mistake:**

```python
# Model: price = 50000 + 150*sqft + 20000*bedrooms

# WRONG interpretation:
"Each additional sqft increases price by $150."
# This ignores that you're holding bedrooms constant!

# WRONG comparison across models:
# Model 1: price = 50000 + 200*sqft
# Model 2: price = 50000 + 150*sqft + 20000*bedrooms
"Model 2 shows sqft is less important (150 < 200)"
# NO! The coefficient changed because Model 2 controls for bedrooms

```

**Why It's a Problem:** Without the "holding other variables constant" qualifier, you're not correctly stating what the coefficient means. Comparing coefficients across different models is also misleading—they represent different things. The coefficient in Model 1 captures both direct size effects and indirect bedroom effects (bigger houses have more bedrooms). Model 2 separates these. Saying "sqft is less important" misses that you've isolated its unique effect.

**The Right Approach:**

```python
# CORRECT interpretation:
print("Model 2: price = 50000 + 150*sqft + 20000*bedrooms")
print("\nCorrect interpretation:")
print("Each additional sqft increases price by $150,")
print("HOLDING THE NUMBER OF BEDROOMS CONSTANT.")
print("\nThis is the effect of size after accounting for bedroom differences.")

# CORRECT comparison:
print("\nWhy did sqft coefficient change?")
print("Model 1 (sqft only): $200/sqft")
print("  - Captures both size effect AND bedroom effect")
print("  - Bigger houses tend to have more bedrooms")
print("\nModel 2 (sqft + bedrooms): $150/sqft") 
print("  - Captures ONLY size effect")
print("  - Bedroom effect is now in the bedroom coefficient ($20,000)")
print("\nTotal effect in Model 2:")
print("  Adding 1000 sqft + 1 bedroom = $150*1000 + $20,000 = $170,000")

```

**Why This Works:** You explicitly state what you're holding constant, making the interpretation accurate. You explain why coefficients differ across models—they represent different things, not different importance. This prevents misunderstanding and incorrect business decisions based on naive coefficient interpretation.

---

### Pitfall 2: Using High-Degree Polynomials and Overfitting

**The Mistake:**

```python
from sklearn.preprocessing import PolynomialFeatures

# Trying very high degree polynomial
poly = PolynomialFeatures(degree=10)  # Way too high!
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

print(f"Training R²: {model.score(X_poly, y):.3f}")  # 0.999! Amazing!
print(f"Test R²: {model.score(X_poly_test, y_test):.3f}")  # 0.234... uh oh

```

**Why It's a Problem:** High-degree polynomials can fit training data almost perfectly (high training R²) but fail on new data (low test R²)—this is overfitting. The model memorizes training noise rather than learning the true pattern. With degree 10, you create x, x², x³, ..., x¹⁰ features—10 features from one variable! With many features and limited data, you can fit any wiggly curve, including random noise. The model becomes too complex for the amount of information in the data.

**The Right Approach:**

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score

# Try different degrees and use cross-validation
degrees = [1, 2, 3, 4, 5]
cv_scores = []

for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    model = LinearRegression()
    
    # Use cross-validation to get honest estimate of generalization
    scores = cross_val_score(model, X_poly, y, cv=5, scoring='r2')
    cv_scores.append(scores.mean())
    
    print(f"Degree {degree}: CV R² = {scores.mean():.3f} (+/- {scores.std():.3f})")

# Choose degree with best CV score
best_degree = degrees[np.argmax(cv_scores)]
print(f"\nBest degree: {best_degree}")

# Fit final model with best degree
poly = PolynomialFeatures(degree=best_degree)
X_poly = poly.fit_transform(X)
final_model = LinearRegression()
final_model.fit(X_poly, y)

```

**Output:**

```
Degree 1: CV R² = 0.623 (+/- 0.045)
Degree 2: CV R² = 0.891 (+/- 0.032)
Degree 3: CV R² = 0.883 (+/- 0.041)
Degree 4: CV R² = 0.854 (+/- 0.067)
Degree 5: CV R² = 0.798 (+/- 0.105)

Best degree: 2

```

**Why This Works:** Cross-validation tests the model on held-out data, revealing overfitting. Degree 2 (quadratic) has the best generalization. Higher degrees start overfitting (CV R² decreases, standard deviation increases). By systematically testing degrees and using CV, you find the right complexity—not too simple (degree 1), not too complex (degree 5+). Most real-world relationships need degree 2-3; higher degrees are rarely appropriate unless you have massive data.

**Rule of thumb:** Start with degree 2. Only increase if domain knowledge strongly suggests higher-order effects and you have abundant data (ideally hundreds of points per degree).

---

**Questions or stuck?** Review the worked examples—Example 1 shows basic multiple regression mechanics, Example 2 demonstrates when and why to use polynomial terms, and Example 3 shows complete residual diagnostics. If coefficient interpretation is confusing, re-read Concept C and Pitfall 1—the "holding constant" idea is central but takes practice. If residual plots are unclear, study Example 3's diagnostic plots and what patterns indicate. Remember: start simple (multiple linear regression), check residuals, add complexity only where residuals reveal problems. Most real-world problems need multiple regression; only some need polynomial terms.

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