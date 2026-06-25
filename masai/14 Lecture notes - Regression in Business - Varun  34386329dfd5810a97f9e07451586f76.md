# 14. Lecture notes - Regression in Business - Varun Raste - 12 Nov 2025

## [In-class Notes](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/91b4328f-822d-4435-b9db-54681b8df341/OnsyChRuojFb3GoF.zip)

# Regression in Business: From Data to Decisions

**What you'll be able to do:**

- Use statsmodels for statistical inference (p-values, confidence intervals, hypothesis tests)
- Forecast business KPIs (revenue, sales, costs) with regression models
- Create and interpret diagnostic plots to validate business models
- Communicate regression results to business stakeholders with statistical rigor

---

## 1. Introduction

### Core Definition

Business regression combines statistical modeling with domain knowledge to forecast Key Performance Indicators (KPIs), test hypotheses about business drivers, and quantify uncertainty in predictions. Unlike machine learning regression (focused on prediction accuracy), business regression emphasizes **statistical inference**—determining which factors are truly significant, quantifying confidence in estimates, and making defensible decisions with known uncertainty. Statsmodels is the primary Python library for this approach, providing rich statistical outputs that sklearn doesn't offer.

### Why This Matters

**Problem it solves:** Business decisions need more than predictions—they need confidence intervals ("revenue will be 1M−1M-1M−1.2M with 95% confidence"), hypothesis tests ("is this marketing channel significantly effective?"), and statistical validation ("can we trust this forecast?"). Sklearn gives predictions; statsmodels gives statistical rigor for decision-making.

**What you'll gain:**

- **Quantified uncertainty:** Not just "sales will be 1000 units" but "1000 ± 50 units (95% CI)"
- **Hypothesis testing:** Answer "does X significantly affect Y?" with p-values
- **Credibility:** Present results with statistical backing that stakeholders trust

**Real-world:** Finance teams use regression to forecast revenue with confidence intervals for budgeting. Marketing teams test which channels drive conversions significantly. Operations teams predict demand to optimize inventory.

---

## 2. Core Concepts

### Concept A: Statsmodels for Statistical Inference

**Definition:** Statsmodels is a Python library focused on statistical modeling and inference. Unlike sklearn (machine learning focus), statsmodels provides p-values, confidence intervals, hypothesis tests, and detailed statistical summaries. It's designed for answering "is this relationship real?" rather than just "what's the prediction?"

**Key features:**

- **P-values:** Test if coefficients are significantly different from zero
- **Confidence intervals:** Range for coefficient estimates (e.g., β₁ = 150 ± 20)
- **Summary tables:** Comprehensive statistical output including R², F-statistic, AIC/BIC
- **Hypothesis tests:** Test specific hypotheses about relationships

**Example:**

```python
import statsmodels.api as sm

# Add constant for intercept
X = sm.add_constant(df[['marketing_spend', 'price']])
y = df['revenue']

# Fit model
model = sm.OLS(y, X).fit()
print(model.summary())

# Output includes:
# - Coefficient estimates
# - Standard errors
# - t-statistics
# - P-values (Pr(>|t|))
# - Confidence intervals
# - R², Adjusted R², F-statistic

```

**Common confusion:** Beginners use sklearn for everything. Use sklearn for prediction-focused ML; use statsmodels when you need statistical inference and hypothesis testing.

---

### Concept B: KPI Forecasting

**Definition:** KPI (Key Performance Indicator) forecasting uses historical data and predictor variables to predict future business metrics—revenue, sales volume, customer acquisition, churn rate, etc. The goal is actionable forecasts with quantified uncertainty, not just point predictions.

**Business KPIs commonly forecasted:**

- **Revenue/Sales:** Predict future revenue based on marketing spend, seasonality, economic indicators
- **Customer metrics:** Acquisition, retention, lifetime value
- **Operational metrics:** Demand, inventory needs, staffing requirements

**Key characteristics:**

- **Time component:** Often includes time trends, seasonality, lagged variables
- **Uncertainty quantification:** Provide prediction intervals, not just point estimates
- **Business constraints:** Must align with business cycles, budget constraints, realistic growth

**Example:**

```python
# Forecast monthly revenue from marketing spend
# revenue = β₀ + β₁*marketing + β₂*month + β₃*month²

model = sm.OLS(revenue, sm.add_constant(features)).fit()

# Future prediction
future_X = sm.add_constant([[50000, 13, 169]])  # $50k marketing, month 13
prediction = model.get_prediction(future_X)

print(f"Predicted revenue: ${prediction.predicted_mean[0]:,.0f}")
print(f"95% CI: ${prediction.conf_int()[0,0]:,.0f} - ${prediction.conf_int()[0,1]:,.0f}")

```

**Remember:** Business forecasts need uncertainty quantification. "Revenue will be 1M±1M ± 1M±100k" is more useful than just "$1M."

---

### Concept C: Diagnostic Plots in Business Context

**Definition:** Diagnostic plots validate model assumptions and reveal problems that could invalidate business conclusions. In business, bad models lead to bad decisions—overforecasting causes overproduction, underforecasting causes stockouts. Diagnostics ensure models are trustworthy.

**Key diagnostic plots:**

- **Residuals vs Fitted:** Check linearity and homoscedasticity
- **Q-Q Plot:** Check normality of residuals (important for confidence intervals)
- **Scale-Location:** Check constant variance
- **Residuals vs Leverage:** Identify influential outliers (e.g., Black Friday spike)

**Business interpretation:**

- Pattern in residuals → model missing something (seasonality, non-linearity)
- Funnel shape → uncertainty increases with forecast size (adjust intervals)
- Influential points → single events (promotions, crises) distorting model

**Example:**

```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 8))

# Residuals vs Fitted
ax1 = fig.add_subplot(221)
ax1.scatter(model.fittedvalues, model.resid)
ax1.axhline(0, color='r', linestyle='--')
ax1.set_xlabel('Fitted values')
ax1.set_ylabel('Residuals')
ax1.set_title('Residuals vs Fitted')

# Q-Q Plot
ax2 = fig.add_subplot(222)
sm.qqplot(model.resid, line='s', ax=ax2)
ax2.set_title('Normal Q-Q')

# Scale-Location
ax3 = fig.add_subplot(223)
ax3.scatter(model.fittedvalues, np.sqrt(np.abs(model.resid_pearson)))
ax3.set_xlabel('Fitted values')
ax3.set_ylabel('√|Standardized residuals|')
ax3.set_title('Scale-Location')

# Residuals vs Leverage
ax4 = fig.add_subplot(224)
sm.graphics.influence_plot(model, ax=ax4, criterion="cooks")
ax4.set_title('Residuals vs Leverage')

plt.tight_layout()

```

**Common confusion:** Skipping diagnostics because R² looks good. A model with R²=0.85 but violated assumptions gives wrong confidence intervals and invalid hypothesis tests.

---

### Concept D: Statistical Significance in Business Decisions

**Definition:** Statistical significance (p-value < 0.05) indicates a relationship is unlikely due to chance. In business, this determines whether to invest in a factor. A significant coefficient means the factor genuinely affects the outcome; non-significant means it might just be noise.

**Interpretation:**

- **p < 0.05:** Factor is statistically significant (conventional threshold)
- **Coefficient sign:** Positive or negative effect
- **Coefficient magnitude:** Size of effect (economic significance)

**Example:**

```python
# Model: sales = β₀ + β₁*tv_ads + β₂*radio_ads

# Output shows:
#              coef    std err    t      P>|t|    [0.025   0.975]
# tv_ads       45.2    5.1       8.86   0.000     35.2     55.2
# radio_ads    12.3    8.4       1.46   0.145     -4.2     28.8

# tv_ads: p=0.000 < 0.05 → statistically significant
# Each $1k in TV ads increases sales by 45.2 units (confident)

# radio_ads: p=0.145 > 0.05 → NOT statistically significant
# Effect might be due to chance (don't rely on this)

```

**Business implication:** Invest more in TV ads (proven effect), reconsider radio ads (no proven effect).

**Remember:** Statistical significance ≠ practical significance. A coefficient might be statistically significant (p<0.05) but economically tiny (adds $0.01 revenue). Check both p-value AND magnitude.

---

## 3. Worked Examples

### Example 1: Basic Statsmodels Analysis

**Scenario:** E-commerce company wants to understand which marketing channels drive sales. They have monthly data on email campaigns, social media ads, and sales.

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Generate realistic business data
np.random.seed(42)
n_months = 24

email_spend = np.random.uniform(5000, 15000, n_months)
social_spend = np.random.uniform(3000, 10000, n_months)
month = np.arange(1, n_months + 1)

# Sales with realistic relationships
sales = (20000 + 
         2.5 * email_spend +      # Strong effect
         0.8 * social_spend +      # Weak effect
         1000 * month +            # Growth trend
         np.random.normal(0, 5000, n_months))

df = pd.DataFrame({
    'month': month,
    'email_spend': email_spend,
    'social_spend': social_spend,
    'sales': sales
})

print("=== Data Summary ===")
print(df.describe())

# Prepare features
X = df[['email_spend', 'social_spend', 'month']]
y = df['sales']

# Add constant (intercept)
X = sm.add_constant(X)

# Fit model
model = sm.OLS(y, X).fit()

# Display summary
print("\n=== Model Summary ===")
print(model.summary())

# Interpret key results
print("\n=== Business Interpretation ===")
for var, coef, pval in zip(['Intercept', 'Email Spend', 'Social Spend', 'Month'],
                           model.params, model.pvalues):
    sig = "✓ Significant" if pval < 0.05 else "✗ Not significant"
    print(f"{var:15s}: Coef={coef:8.2f}, p-value={pval:.4f} {sig}")

print(f"\nModel R²: {model.rsquared:.3f}")
print(f"Adjusted R²: {model.rsquared_adj:.3f}")

# Confidence intervals
print("\n=== 95% Confidence Intervals ===")
conf_int = model.conf_int()
for var, (lower, upper) in zip(['Intercept', 'Email', 'Social', 'Month'], 
                                conf_int.values):
    print(f"{var:10s}: [{lower:8.2f}, {upper:8.2f}]")

```

**Output:**

```
=== Model Summary ===
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  sales   R-squared:                       0.887
Model:                            OLS   Adj. R-squared:                  0.870
...
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       2.012e+04   4521.234      4.450      0.000    1.07e+04    2.95e+04
email_spend    2.4823      0.245     10.131      0.000       1.973       2.992
social_spend   0.7912      0.354      2.235      0.037       0.055       1.527
month       1012.3456     89.234     11.345      0.000     826.234    1198.457
==============================================================================

=== Business Interpretation ===
Intercept      : Coef=20120.00, p-value=0.0002 ✓ Significant
Email Spend    : Coef=    2.48, p-value=0.0000 ✓ Significant
Social Spend   : Coef=    0.79, p-value=0.0368 ✓ Significant
Month          : Coef= 1012.35, p-value=0.0000 ✓ Significant

Model R²: 0.887
Adjusted R²: 0.870

=== 95% Confidence Intervals ===
Intercept : [10700.00, 29540.00]
Email     : [    1.97,     2.99]
Social    : [    0.06,     1.53]
Month     : [  826.23,  1198.46]

```

**Business insights:**

1. **Email is strongest driver:** Each 1inemailspendadds1 in email spend adds 1inemailspendadds2.48 in sales (p<0.001, highly significant)
2. **Social media works but weaker:** Each 1insocialadds1 in social adds 1insocialadds0.79 in sales (p=0.037, significant but close to threshold)
3. **Growth trend:** Sales increase by ~$1000/month naturally (market growth)
4. **Recommendation:** Prioritize email marketing (better ROI: 2.48x vs 0.79x)

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