# 33. Lecture notes - Fraud Detection in Finance - Varun Raste - 2 Jan 2026

## [In-class resources](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/0c9640b5-bc13-417a-a975-e6563eb8b1fd/03wtZ3HZGGVCFJQk.zip)

# Fraud Detection in Finance: Pre-Read

**Prerequisites:** Familiarity with basic ML concepts (training, testing, metrics), Python/Pandas, and understanding of anomaly detection fundamentals.

---

## What You'll Gain from This Pre-Read

**After reading, you'll be able to:**

- Recognize the key features that power real-world fraud detection systems
- Understand why Isolation Forest is popular for fraud and how to tune it
- Explain how scoring thresholds translate into business decisions (approve/review/decline)
- Identify monitoring signals that indicate your fraud model is degrading

**Think of this as:** Learning the playbook before the game—you'll understand the strategy even if you haven't run the plays yet.

---

## The Big Picture: Why Does This Matter?

Every second, payment networks process thousands of transactions. Hidden among legitimate purchases are fraudsters testing stolen cards, account takeover attempts, and organized fraud rings exploiting system weaknesses. Card issuers, BNPL (Buy Now Pay Later) apps, and payment processors lose billions annually—and that cost ultimately passes to consumers and merchants.

The challenge? You have milliseconds to decide. Labels arrive days or weeks later (when chargebacks come in), and fraud patterns evolve constantly. Your models must detect anomalies in real-time using handcrafted features, tuned detection algorithms, and smart threshold rules—all while minimizing false positives that frustrate legitimate customers.

**Where You'll Use This:**

**Job roles:** Fraud data scientist building detection models; risk analyst tuning thresholds and investigating cases; ML engineer deploying real-time scoring systems; product manager defining the customer experience around step-up authentication.

**Real products:** Stripe Radar scoring every transaction in real-time; PayPal's risk engine analyzing behavioral patterns; Square detecting anomalous merchant activity; credit card authorization systems making instant approve/decline decisions.

**What you can build:** Real-time risk scoring APIs, step-up authentication triggers (like 2FA challenges), case management queues for human review, fraud ring detection systems, velocity monitoring dashboards.

**Think of it like this:** You are a bouncer at a crowded concert venue. You cannot interview every attendee, so you watch for tells—nervous behavior, mismatched IDs, people who don't fit the pattern. When signals stack up, you escalate. Fraud detection encodes these "tells" as features and uses algorithms to make consistent, millisecond decisions at scale.

**Limitation:** Unlike a bouncer who can ask questions, your model only sees the data you've engineered—it cannot request clarification or use intuition about context it wasn't trained on.

---

## Your Roadmap Through This Topic

**1. Feature Engineering for Fraud**
You'll learn to transform raw payment logs into powerful predictive signals: velocity counts (how many transactions in the last 10 minutes?), geolocation distance (is this physically possible?), device fingerprints (have we seen this browser before?), merchant risk tiers, and behavior baselines. These features are often more important than the algorithm itself.

**2. Isolation Forest (IF) Tuning**
You'll explore how Isolation Forest works by randomly partitioning data—anomalies get isolated quickly because they're different from the crowd. You'll understand the key hyperparameters: `contamination` (expected fraud rate) and `max_samples` (how much data each tree sees), and learn strategies for validation when labeled data is scarce.

**3. Scoring and Threshold Rules**
You'll design action bands that translate raw anomaly scores into business decisions: green (approve instantly), amber (trigger step-up authentication), red (decline and open investigation). You'll learn to use cost matrices to balance false positives (blocking good customers) against false negatives (missing fraud).

**4. Monitoring and Drift Checks**
You'll discover how fraud models degrade over time as attackers adapt and customer behavior shifts. Key monitoring signals include score distribution drift, false decline rate spikes, and feature decay. You'll learn when to retrain versus when to simply recalibrate thresholds.

**5. Feedback Loops and Label Collection**
You'll understand the challenge of delayed labels (chargebacks arrive 30-90 days later) and how to design feedback loops that improve your model over time without waiting months for ground truth.

---

## Key Terms to Listen For

**Velocity feature:** Count of recent actions (e.g., transactions in past 10 minutes) that captures bursty fraud behavior.
*Example:* A card that suddenly has 5 transactions in 2 minutes after months of 1-2 per day is suspicious.

**Distance feature:** Geographic distance between consecutive events, often using Haversine formula for lat/long coordinates.
*In practice:* A transaction in New York 20 minutes after one in Los Angeles is physically impossible—this "impossible travel" pattern is a strong fraud signal.

**Device fingerprint:** A stable identifier built from browser user agent, screen resolution, timezone, and hardware signals that links related events.
*Think of it as:* A digital "thumbprint" that persists even when someone clears cookies.

**Isolation Forest contamination:** The expected proportion of outliers in your dataset—controls how aggressively the model flags points as anomalies.
*Tuning tip:* Set this close to your known fraud rate; too high causes false positives, too low misses fraud.

**Anomaly score threshold:** The cutoff value on Isolation Forest scores that maps to action bands (approve/review/decline).
*In practice:* Scores above 0.6 might approve, 0.6-0.8 trigger OTP verification, above 0.8 auto-decline.

**Cost matrix:** Business costs assigned to each outcome type—used to optimize thresholds beyond raw accuracy.
*Example:* If a false decline costs 50inlostrevenueandbranddamage,butamissedfraudcosts50 in lost revenue and brand damage, but a missed fraud costs 50inlostrevenueandbranddamage,butamissedfraudcosts200, you'd bias toward approval.

**Step-up authentication:** Additional verification (like OTP or security questions) triggered for medium-risk transactions instead of outright decline.
*Why it matters:* Lets you challenge suspicious transactions without blocking legitimate customers.

---

## Example: Scoring an Unusual Transaction in the Real World

**The scenario:** A payment platform sees a 1,200electronicspurchaseonacardthattypicallybuys1,200 electronics purchase on a card that typically buys 1,200electronicspurchaseonacardthattypicallybuys80 groceries weekly. The transaction is from a new IP address in a different state.

**The features computed:**

- **Velocity:** 0 transactions in past 24 hours (normal for this user)
- **Amount deviation:** 15x the cardholder's average ticket size (highly unusual)
- **Merchant risk tier:** Electronics category (higher fraud risk than groceries)
- **Distance from last IP:** 800 miles from previous login (suspicious but possible if traveling)
- **Device fingerprint:** Never seen before for this user (concerning)

**The model decision:**

```python
# Features assembled into a vector
features = [velocity_24h, amount_zscore, merchant_risk, distance_miles, new_device_flag]

# Isolation Forest produces anomaly score
iso_forest = IsolationForest(contamination=0.03, max_samples=0.7)
score = iso_forest.decision_function([features])[0]
# Lower (more negative) scores = more anomalous

# Map to action bands
if score > -0.2:
    action = "APPROVE"      # Green band
elif score > -0.5:
    action = "CHALLENGE"    # Amber band - trigger OTP
else:
    action = "DECLINE"      # Red band - block and investigate

```

**The outcome:** The transaction lands in the amber band. The system triggers an OTP challenge. The legitimate cardholder would enter the code; in this case, OTP verification fails—confirming fraud. The transaction is declined before shipment, saving $1,200 without blocking legitimate customers who would have passed the challenge.

**Key takeaway:** The scoring system doesn't just approve/decline—the middle "challenge" band lets you verify suspicious transactions without frustrating legitimate customers.

---

## Questions to Keep in Mind

1. 
**Which 3-5 features would you compute first if labels are limited and latency must stay under 100ms?**
Consider: Which features give the most signal with the least computation?

2. 
**How should thresholds shift between a low-risk debit card portfolio and a high-risk cross-border marketplace?**
Think about: What's the cost of a false decline vs. a missed fraud in each context?

3. 
**What monitoring signals tell you that your current scoring rules are causing too many false declines or letting fraud through?**
Consider: How would you detect that attackers have adapted to your model?

**Reflect:** If you were building a fraud system from scratch tomorrow, what's the first feature you'd compute and why?

---

## Quick Self-Check: Did It Click?

After reading, you should be able to:

- Name at least 3 features commonly used in fraud detection (velocity, distance, device fingerprint, etc.)
- Explain why Isolation Forest works well for fraud (rare events get isolated quickly)
- Describe what "action bands" are and why they're better than binary approve/decline
- Identify why fraud models need continuous monitoring

**If you checked fewer than 3 boxes:** Focus on the Key Terms section—understanding the vocabulary is key to following the session.

---

## How This Topic Connects

**Builds on:** Anomaly detection fundamentals, basic feature engineering, understanding of precision/recall tradeoffs.

**Enables:**

- Building production-grade fraud scoring APIs
- Designing cost-aware decision systems
- Creating monitoring dashboards for ML model health

**Related concepts you might explore:**

- Graph-based fraud detection (finding fraud rings through transaction networks)
- Behavioral biometrics (using typing patterns and mouse movements as features)
- Explainable AI for fraud (telling users why a transaction was declined)

---

**You're ready to dive deeper into Fraud Detection in Finance!** This foundation will help you understand the technical decisions that protect millions of transactions every day.

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