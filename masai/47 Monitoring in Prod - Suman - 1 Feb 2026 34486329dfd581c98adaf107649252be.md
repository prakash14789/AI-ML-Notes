# 47. Monitoring in Prod - Suman - 1 Feb 2026

# Monitoring in Production: Prometheus, Grafana, Model Drift Alerts, and Retrain Triggers

# [In-Class Notes](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/8ed8d076-0b0a-4b76-b54a-cf936ef12d5a/JUpLdguA0w0EDN0n.zip)

**Prerequisites:** Understanding of ML model deployment, basic knowledge of HTTP APIs, familiarity with metrics concepts (latency, throughput)

**What you'll be able to do:**

- Configure Prometheus to scrape ML model metrics
- Build Grafana dashboards for model observability
- Implement drift detection and alerting for production models
- Design retrain triggers based on drift thresholds or performance degradation

---

## 1. Introduction: What is ML Monitoring and Why Should You Care?

### Core Definition

ML monitoring extends traditional application monitoring to track model-specific health indicators: prediction distributions, feature drift, model performance metrics, and data quality. While infrastructure monitoring asks "Is the service running?", ML monitoring asks "Is the model still making good predictions?"

This requires instrumenting models to export ML-specific metrics, storing those metrics in time-series databases like Prometheus, visualising trends in Grafana, and alerting when metrics indicate degradation.

### A Simple Analogy

Traditional monitoring is like a heart rate monitor—it tells you the patient is alive. ML monitoring is like a full diagnostic panel—blood pressure, cholesterol, oxygen levels. The heart might beat steadily while other indicators silently worsen. You need the full picture to catch problems early.

**Limitation:** Unlike medical diagnostics with clear normal ranges, ML metrics often require establishing baselines from your specific model and data—there's no universal "healthy" threshold.

### Why This Matters to You

**Problem it solves:** Models degrade silently in production. Without ML-specific monitoring, you discover problems from angry customers, not dashboards.

**What you'll gain:**

- Early warning systems that catch drift before it impacts users
- Data-driven decisions about when to retrain models
- Confidence that your models are performing as expected in production

**Real-world context:** Companies like Netflix, Uber, and Airbnb have dedicated ML platform teams whose primary job is monitoring model health. This isn't optional for production ML—it's essential.

---

## 2. The Foundation: Core Concepts Explained

### Concept A: Prometheus Architecture

**Definition:** Prometheus is a pull-based monitoring system. Your applications expose metrics at an HTTP endpoint (typically `/metrics`), and Prometheus periodically scrapes these endpoints, storing the data as time series with timestamps and labels.

**Key characteristics:**

- Pull-based: Prometheus fetches metrics rather than receiving pushed data
- Multi-dimensional: Metrics have labels enabling flexible querying
- Local storage: Time-series data stored efficiently on disk
- PromQL: Powerful query language for aggregating and analysing metrics

**A concrete example:**

```python
# Your model exposes metrics at http://model-service:8000/metrics
# Prometheus scrapes this endpoint every 15 seconds

# Example metric output:
# model_predictions_total{model="churn_v1",outcome="positive"} 1523
# model_predictions_total{model="churn_v1",outcome="negative"} 8477
# prediction_latency_seconds_bucket{le="0.1"} 9500

```

**Common confusion:** Beginners expect to push metrics to Prometheus like a logging system. Instead, you expose metrics and let Prometheus pull them—this simplifies service discovery and reduces coupling.

---

### Concept B: Types of Model Drift

**Definition:** Drift occurs when something changes between training time and inference time, causing model performance to degrade. There are three primary types, each requiring different detection strategies.

**How it relates to Prometheus:** You'll export drift scores as Prometheus gauges, updated periodically by comparing recent production data against training data reference distributions.

**Key characteristics:**

- **Data Drift (Covariate Shift):** Input feature distributions change
- **Concept Drift:** The relationship between features and target changes
- **Prediction Drift:** Model output distribution shifts

**A concrete example:**

Drift Type | Example | Detection Method
Data Drift | Average customer age shifts from 35 to 45 | Compare feature distributions (PSI, KS test)
Concept Drift | Fraudsters adopt new tactics | Monitor performance metrics when labels arrive
Prediction Drift | Model suddenly predicts 80% positive instead of 50% | Track prediction distribution over time

**Remember:** Prediction drift is often the earliest signal because it doesn't require ground truth labels. Data drift explains why predictions changed. Concept drift explains why accurate predictions became wrong.

---

### How Prometheus Metrics and Drift Detection Work Together

Prometheus stores time-series drift scores, enabling you to track how drift evolves over days or weeks. Grafana visualises these trends. Alertmanager fires alerts when drift exceeds thresholds. Together, they create a closed-loop system that detects degradation and triggers responses.

---

## 3. Seeing It in Action: Worked Example

**Tip:** This example connects all concepts—metrics, visualisation, and alerting—in a realistic scenario.

### Complete Monitoring Setup for an ML Model

**Scenario:** You're running a loan default prediction model. You need to monitor latency, prediction distribution, and feature drift. When drift exceeds thresholds, you want automatic alerts.

**Our approach:** Instrument the model with prometheus-client, configure Prometheus to scrape metrics, build a Grafana dashboard, and set up alerting rules.

**Step-by-step solution:**

```python
# model_service.py - Instrumented FastAPI model service
from fastapi import FastAPI
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import numpy as np

app = FastAPI()

# Define metrics
predictions_total = Counter(
    'loan_predictions_total',
    'Total loan predictions',
    ['prediction']  # Label: approved/denied
)

latency_seconds = Histogram(
    'loan_prediction_latency_seconds',
    'Prediction latency in seconds',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
)

drift_psi = Gauge(
    'loan_feature_drift_psi',
    'Population Stability Index for features',
    ['feature_name']
)

prediction_mean = Gauge(
    'loan_prediction_mean',
    'Rolling mean of prediction probabilities'
)

# Inference endpoint with instrumentation
@app.post("/predict")
@latency_seconds.time()
def predict(features: dict):
    probability = model.predict_proba(features)[0][1]
    prediction = "approved" if probability > 0.5 else "denied"

    predictions_total.labels(prediction=prediction).inc()
    update_rolling_stats(probability)

    return {"prediction": prediction, "probability": probability}

# Metrics endpoint for Prometheus
@app.get("/metrics")
def metrics():
    return generate_latest()

# Background task: Calculate drift every hour
def calculate_drift():
    for feature in monitored_features:
        psi = compute_psi(reference_data[feature], recent_data[feature])
        drift_psi.labels(feature_name=feature).set(psi)

```

**Prometheus configuration (prometheus.yml):**

```yaml
scrape_configs:
  - job_name: 'loan-model'
    scrape_interval: 15s
    static_configs:
      - targets: ['model-service:8000']

```

**Alerting rules (alerts.yml):**

```yaml
groups:
  - name: model-drift-alerts
    rules:
      - alert: HighFeatureDrift
        expr: loan_feature_drift_psi > 0.2
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "Feature drift detected: {{ $labels.feature_name }}"

      - alert: PredictionDistributionShift
        expr: abs(loan_prediction_mean - 0.35) > 0.1
        for: 30m
        labels:
          severity: critical
        annotations:
          summary: "Prediction distribution has shifted significantly"

```

**What just happened:** The model exports four metric types. Prometheus scrapes them every 15 seconds. Alert rules evaluate continuously—if PSI exceeds 0.2 for 10 minutes, an alert fires. The prediction mean alert catches when outputs drift from the expected 35% approval rate.

**Check your understanding:** Why do we use `for: 10m` in the alert rule instead of alerting immediately?

---

## 4. Common Pitfalls: What Can Go Wrong

- **The Mistake:** Only monitoring infrastructure metrics (CPU, memory, latency)

**Why It's a Problem:** A model can return predictions in 10ms with 0% errors while being completely wrong
**The Right Approach:** Monitor prediction distributions, drift scores, and downstream business metrics alongside infrastructure
**Why This Works:** ML failures are semantic, not syntactic—the model returns valid responses that happen to be incorrect

---

- **The Mistake:** Setting drift thresholds too tight

**Why It's a Problem:** Alert fatigue—you get paged for natural variation and start ignoring alerts
**The Right Approach:** Establish baselines during normal operation, set thresholds at 2-3 standard deviations from normal variation
**Why This Works:** Separates true anomalies from expected fluctuation

---

- **The Mistake:** Retraining immediately when drift is detected

**Why It's a Problem:** Drift might be temporary (holidays, promotions), and retraining is expensive
**The Right Approach:** Investigate drift causes first; implement tiered responses (alert → investigate → retrain)
**Why This Works:** Some drift is expected and acceptable; only persistent or severe drift warrants retraining

**If you're stuck:** Start with prediction distribution monitoring—it's the simplest metric that catches most problems.

---

## 5. Your Turn: Practice & Self-Assessment

### Practice Task (Estimated: 20 minutes)

**The Challenge:** Design a monitoring configuration for a fraud detection model that processes 10,000 transactions per minute.

**Specifications:**

- Define at least 3 Prometheus metrics (use appropriate metric types)
- Write one PromQL query that calculates the 5-minute average fraud rate
- Create an alert rule that fires when the fraud rate exceeds 2x the historical average
- Explain when you would trigger a retrain

**Hint:** Think about what metrics would help you answer: "Is the model catching fraud?", "Is the model seeing different data than before?", and "Is the model's behavior changing?" Match each question to a metric type.

**Extension (optional):** Design a Grafana dashboard layout with 4 panels that would give an on-call engineer complete visibility into model health.

---

### Check Your Understanding

1. **Explanation question:** Why is prediction drift often detectable before data drift or concept drift?
2. **Application question:** You see high PSI on the "income" feature but model accuracy hasn't dropped. Should you retrain?
3. **Error analysis:** An alert fires every Monday morning and auto-resolves by afternoon. What's likely happening and how would you fix it?
4. **Transfer question:** How would you apply these monitoring principles to a recommendation system instead of a classification model?

**Answers & Explanations:**

1. Prediction drift only requires model outputs, available immediately. Data drift requires comparing feature distributions. Concept drift requires ground truth labels, which may be delayed by days or weeks.
2. Not necessarily. If accuracy is stable, the model may be robust to this drift. Investigate whether the drift is within the feature ranges seen during training. Monitor but don't retrain without evidence of degradation.
3. Likely a weekly pattern (different Monday traffic). Fix by using time-aware baselines that compare Monday to previous Mondays, not to the overall average.
4. Monitor recommendation diversity, click-through rate distribution, and user engagement metrics. Drift detection applies to user behavior features and recommendation output distributions.

---

## 6. Consolidation: Key Takeaways & Next Steps

### The Essential Ideas

- **Monitor the model, not just infrastructure:** Latency and uptime don't tell you if predictions are correct
- **Understand drift types:** Data drift (inputs changed), concept drift (relationships changed), prediction drift (outputs changed)
- **Alert thoughtfully:** Baselines, appropriate thresholds, and tiered responses prevent alert fatigue

### Mental Model Check

By now, you should think of production models as living systems that require continuous observation. Training is not the end—it's the beginning of the model's lifecycle. Monitoring tells you when that lifecycle needs intervention.

### What You Can Now Do

You can build comprehensive observability for production ML systems—from metric instrumentation through alerting to retrain decisions. This skill is essential for any production ML role.

### Next Steps

**To deepen this knowledge:** Deploy a model with full Prometheus/Grafana monitoring and intentionally introduce drift to see your alerts fire.

**To build on this:** Learn about feature stores for maintaining consistent features between training and inference, reducing data drift at the source.

**Additional resources:** Prometheus documentation for PromQL, Grafana alerting best practices, Google's paper on ML system technical debt.

---

## Quick Reference Card

Metric Type | Use Case | Example
Counter | Totals that only increase | Total predictions, errors
Gauge | Values that go up and down | Drift score, queue size
Histogram | Distribution of values | Latency, prediction probabilities

Drift Type | Detection | Response Time
Data Drift | PSI, KS test on features | Immediate
Prediction Drift | Output distribution monitoring | Immediate
Concept Drift | Performance metrics with labels | Delayed (needs labels)

**Alert Rule Template:**

```yaml
alert: <AlertName>
expr: <PromQL expression>
for: <duration before firing>
labels:
  severity: <warning|critical>
annotations:
  summary: "<Human-readable description>"

```

---

**Questions or stuck?** Start with prediction distribution monitoring—if you track one thing, track your model's outputs.

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