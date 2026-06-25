# 32. Streaming Anomaly Pipelines - Varun Raste - 2 Jan 2026

### In-Class Resources:

[Isolation_forest_tSNE_PCA_Kafka-main](https://drive.google.com/file/d/12dqGvkj1x5uAjJftOnOxoV98f6wpU2Gu/view?usp=drive_link)

# Streaming Anomaly Pipelines: Kafka, Windowed Stats, and Alert Thresholds

**Prerequisites:** Basics of publish/subscribe messaging, JSON events, and simple Python dict/list handling.

**Time to complete:** 35-45 minutes

**What you'll be able to do:**

- Explain how Kafka partitions, consumer groups, and offsets keep anomaly jobs reliable
- Choose window types and thresholds that balance detection speed with alert noise

---

## 1. Introduction: What is a Streaming Anomaly Pipeline and Why Should You Care?

### Core Definition

A streaming anomaly pipeline ingests events continuously, builds short-lived statistics over time windows, and flags deviations within seconds. Kafka buffers the stream, a processor computes windowed metrics, and thresholds turn those metrics into alerts.

[Kafka-Powered Anomaly Flow]

### A Simple Analogy

A toll plaza counts cars every minute. A sliding total over the last five minutes is "normal." A sudden spike or drop signals trouble. The analogy captures rolling baselines but not Kafka's partition ordering or replay controls.

### Why This Matters to You

- **Problem it solves:** Batch checks surface incidents hours late.
- **What you'll gain:** Confidence to wire Kafka topics, choose windows, and tune thresholds without alert fatigue.
- **Where it shows up:** Payments fraud, API SLOs, IoT drift, SOC log triage.

---

## 2. The Foundation: Core Concepts Explained

### Concept A: Kafka as the Streaming Backbone

Kafka is an append-only log split into partitions. Producers write; consumer groups read ordered offsets, enabling scale and replay. Partitions scale throughput; order holds inside each partition. Consumer groups share work; offsets/checkpoints let you replay after failures. Example: a 6-partition `transactions` topic with a 3-consumer `fraud-detector` group gives each instance two partitions-key by `user_id` to keep related events together.

### Concept B: Windowed Aggregations

Windowing groups events into time (or count) buckets to compute rolling baselines.

**Tumbling:** fixed, non-overlapping windows (e.g., each minute).

**Sliding:** overlapping windows (size 5 min, hop 1 min) for smoother signals.

**Session:** close when inactivity exceeds a gap (e.g., 60s).

Short windows = faster/noisier; long windows = slower/smoother.

### Concept C: Alert Thresholds

Thresholds convert windowed metrics into alerts-either static (`latency_ms > 400`) or adaptive (mean/median +/- deviation). Add hysteresis: consecutive breaches or cool-downs to prevent flapping. Example: alert if 5-minute sliding failure rate exceeds median + 3xMAD for two hops in a row. Tune window size, k, and breach count together.

### How These Concepts Work Together

Kafka supplies ordered events, windowing turns them into baselines, thresholds page humans or automation. Think conveyor belt (Kafka) -> measuring station (windows) -> traffic light (thresholds).

---

## 3. Seeing It in Action: Worked Example

**Scenario:** A Kafka topic `api-latency` emits `{service, latency_ms, timestamp}`. You want alerts when latency spikes relative to recent behavior.

**Approach:** Use a 5-minute sliding window with 1-minute hop. Compute mean and std per service. Alert when `latency_ms > mean + 3*std` for two consecutive hops.

```python
WINDOW, HOP = timedelta(minutes=5), timedelta(minutes=1)
windows = defaultdict(list)
breaches = defaultdict(int)

for msg in KafkaConsumer("api-latency", group_id="latency-detector",
                         enable_auto_commit=False,
                         value_deserializer=lambda v: json.loads(v)):
    ev = msg.value
    ts, svc = datetime.fromisoformat(ev["timestamp"]), ev["service"]

    # assign to overlapping windows
    for i in range(int(WINDOW / HOP)):
        start = ts - i * HOP
        if ts < start + WINDOW:
            windows[(svc, start)].append(ev["latency_ms"])

    # evaluate latest window
    vals = windows[(svc, ts - (ts % HOP))]
    if len(vals) >= 5 and vals[-1] > statistics.mean(vals) + 3 * (statistics.pstdev(vals) or 1):
        breaches[svc] += 1
        if breaches[svc] >= 2:
            send_alert(svc, vals[-1])
    else:
        breaches[svc] = 0

```

**What happened:** Ordered delivery + sliding windows + z-score with hysteresis detect spikes within ~1-2 minutes.

---

## 4. Common Pitfalls: What Can Go Wrong and How to Avoid It

- **Using processing time instead of event time:** Late or out-of-order events fall into wrong windows. Use event timestamps with watermarking/allowed lateness.
- **Static thresholds on drifting traffic:** Fixed numbers page constantly as volume grows or miss regressions. Prefer median/mean + deviation plus hysteresis and cool-downs.

---

## 5. Your Turn: Practice & Self-Assessment

### Practice Task (15-20 minutes)

Build a script that reads `checkout-events` from Kafka, applies a 10-minute sliding window (hop 2 minutes) on `failure_rate` per `region`, and posts to a webhook when rate exceeds median + 3xMAD for three hops in a row.

**Must-haves:** event time with 90 seconds allowed lateness; dry-run flag; 10-minute cool-down per region unless failure_rate doubles. Write a small `median_mad(values)` helper before wiring the webhook client.

### Check Your Understanding

1. Why do sliding windows smooth anomaly scores compared to tumbling windows?
2. Spiky alerts follow every deploy. What knob do you adjust first-window size, threshold multiplier, or lateness allowance?

**Answers (brief):** Overlap smooths scores; widen window/hop before raising the multiplier unless lateness is obvious.

### Self-Assessment

- Pick window types for a new metric
- Implement a statistical threshold with hysteresis
- Translate window + threshold settings into expected detection latency

If you missed any, redo the practice task and log how each parameter changes alert volume.

---

## 6. Consolidation: Key Takeaways & Next Steps

- Kafka provides replayable, ordered events; design keys so related events stay together.
- Window type and size set the balance between speed and noise.
- Adaptive thresholds plus hysteresis reduce alert fatigue while catching real incidents.

**Next steps:** Test the practice task on a sample Kafka topic, then try declarative windowed alerts in ksqlDB or Flink SQL.

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