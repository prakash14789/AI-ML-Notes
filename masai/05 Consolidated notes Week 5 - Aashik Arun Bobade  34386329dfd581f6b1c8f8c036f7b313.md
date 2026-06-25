# 05. Consolidated notes: Week 5 - Aashik Arun Bobade - 11 Sep 2025

## 1. Calculus Basics

- **Derivatives**: Rate of change of a function. Example: slope of a curve.
- **Gradients**: Vector of partial derivatives (multi-variable functions).
- **SymPy**: Python library to symbolically compute derivatives, integrals, simplify equations.

---

## 2. Descriptive Statistics

- **Mean**: Average value.
- **Median**: Middle value.
- **Mode**: Most frequent value.
- **Variance/Std. Dev.**: Spread of data.

---

## 3. Probability Distributions

- **Normal (Gaussian)**: Bell curve, most data near mean.
- **Binomial**: Discrete yes/no outcomes (coin flips).
- **Poisson**: Rare events (calls per hour).

---

## 4. Central Limit Theorem (CLT)

- If you take many samples, the sample mean follows a **normal distribution**, no matter the original data.
- Basis for hypothesis testing, confidence intervals.

---

## 5. Auto-Diff for ML

- **Numerical derivative**: Approximate slope by tiny difference → less accurate.
- **Analytic derivative**: Exact formula → accurate.
- **PyTorch autograd**: Automatically computes gradients for deep learning models.
- **Gradient checks**: Compare numerical vs analytic to validate correctness.

---

## 6. Statistical Testing Toolbox

- **SciPy stats**: Functions for t-test, chi-square, ANOVA, etc.
- **Bootstrap**: Resample data to estimate variability and confidence intervals.
- **Power analysis**: Check sample size needed to detect an effect with high probability.

---

## 7. Key Takeaways

- Calculus (derivatives/gradients) underpins optimization in ML.
- Descriptive stats + probability distributions build intuition for data.
- CLT explains why normal distribution is everywhere.
- Autograd (PyTorch) automates gradient computation.
- SciPy & bootstrap help with real-world statistical testing.

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