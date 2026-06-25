# 71. Guardrails & Monitoring in AI Applications - Suman - 25 Apr 2026

# Lecture Notes: Guardrails & Monitoring in AI Applications

---

## 1. What You'll Learn

In this lesson, you'll learn to:

- **Build** an input/output guardrail pipeline using the OpenAI Moderation API
- **Apply** regex filters to catch predictable harmful or off-topic content
- **Set up** structured logging to record every AI interaction
- **Configure** threshold-based alerts that notify you when something goes wrong

---

## 2. Detailed Explanation

### Introduction

In the pre-read, you saw what guardrails *are*. Now we go deeper — we build them.

Think of your AI app as a busy restaurant kitchen. The **OpenAI Moderation API** is the health inspector at the door. **Regex filters** are the kitchen rules posted on the wall ("no shellfish in this area"). **Logging** is the order book. **Alerts** are the fire alarm. Together, they make the kitchen safe and auditable.

A production AI app without these layers is not just risky — it is unprofessional.

---

### Why It Matters

**Real problems that guardrails prevent:**

- A mental health chatbot gives a distressed user dangerous advice — because no output filter checked the reply.
- A tutoring app leaks a student's name and school in a response — because no PII (personally identifiable information) filter was in place.
- A support bot starts hallucinating wrong product prices for three days — and nobody notices because there are no logs.

These are not edge cases. They happen in real deployments every week.

---

### Part 1 — OpenAI Moderation API

The **OpenAI Moderation API** is a free, fast classifier. You send it any string of text. It returns a `flagged` boolean and scores for 11 harm categories.

**Categories it detects:**

- `hate`, `hate/threatening`
- `harassment`, `harassment/threatening`
- `self-harm`, `self-harm/intent`, `self-harm/instructions`
- `sexual`, `sexual/minors`
- `violence`, `violence/graphic`

**How to use it:**

```python
import openai

def moderate(text):
    result = openai.moderations.create(input=text)
    output = result.results[0]
    return output.flagged, output.categories

```

Call this on **both the user's input and the model's output**. Most developers only check input — that is a mistake. The model can still generate harmful content on its own.

**Common mistake:** Checking moderation only on input.
**Fix:** Always run moderation on the model's reply too, before displaying it.

---

### Part 2 — Regex Filters

The Moderation API is great for harmful content. But it does not catch everything. What if you want to block competitor names? Prevent users from sharing phone numbers? Stop off-topic requests?

That is where **regex** comes in.

**Regex** (regular expression) is a pattern that matches text. Python's `re` module handles it.

```python
import re

BLOCKED_PATTERNS = [
    r"\b(competitor_name)\b",        # competitor mentions
    r"\b\d{10}\b",                   # 10-digit phone numbers
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # email addresses
]

def has_policy_violation(text):
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False

```

**When to use regex vs. moderation API:**

Situation | Use
Hate speech, violence, adult content | Moderation API
Specific words, phone numbers, emails, competitor names | Regex
Both | Run both — they complement each other

**Common mistake:** Writing a regex that is too broad (e.g., blocking the word "kill" flags "skill" and "toolkit").
**Fix:** Use word boundaries — `\bkill\b` — so it only matches the whole word.

---

### Part 3 — Logging

**Logging** means writing a timestamped record of every important event in your app. Python's built-in `logging` module makes this straightforward.

**What to log:**

- The user's original input
- The model's raw output
- Whether moderation flagged anything
- Response time (how long the API call took)
- Any errors

```python
import logging

logging.basicConfig(
    filename="ai_app.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

def log_interaction(user_input, model_output, flagged):
    logging.info(f"INPUT: {user_input[:100]}")
    logging.info(f"OUTPUT: {model_output[:100]}")
    logging.info(f"FLAGGED: {flagged}")

```

**Log levels to know:**

Level | When to use
INFO | Normal events — every request
WARNING | Something odd but not broken
ERROR | Something failed — catch this in alerts

**Common mistake:** Logging everything at `DEBUG` level in production. Logs grow huge and become unreadable.
**Fix:** Use `INFO` for interactions, `WARNING` for flags, `ERROR` for failures.

---

### Part 4 — Alerts

Logs tell you what happened. **Alerts** tell you right now that something needs attention.

A simple alert strategy uses a counter. If the flagged message count exceeds a threshold in a time window, trigger a notification.

```python
from collections import deque
from datetime import datetime, timedelta

flag_times = deque()

def check_alert_threshold(threshold=5, window_minutes=10):
    now = datetime.now()
    cutoff = now - timedelta(minutes=window_minutes)
    while flag_times and flag_times[0] < cutoff:
        flag_times.popleft()
    if len(flag_times) >= threshold:
        send_alert(f"ALERT: {len(flag_times)} flags in last {window_minutes} mins!")

def record_flag():
    flag_times.append(datetime.now())
    check_alert_threshold()

def send_alert(message):
    # In production: send to Slack, email, PagerDuty, etc.
    print(f"[ALERT] {message}")

```

**Real-world alert destinations:** Slack webhooks, email via SendGrid, PagerDuty for on-call teams.

---

### Putting It All Together

Here is the complete guardrail pipeline:

```python
def safe_chat(user_input):
    # Step 1 — Check input
    flagged_in, _ = moderate(user_input)
    if flagged_in or has_policy_violation(user_input):
        log_interaction(user_input, "BLOCKED", True)
        record_flag()
        return "I can't help with that request."

    # Step 2 — Call the model
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": user_input}]
    )
    reply = response.choices[0].message.content

    # Step 3 — Check output
    flagged_out, _ = moderate(reply)
    if flagged_out:
        log_interaction(user_input, "OUTPUT BLOCKED", True)
        record_flag()
        return "Something went wrong. Please try again."

    # Step 4 — Log and return
    log_interaction(user_input, reply, False)
    return reply

```

Every user message flows through: **moderate input → call model → moderate output → log → return**.

---

### Common Mistakes Summary

Mistake | Why It's a Problem | Fix
Only moderating input | Model can still generate harmful output | Moderate output too
No logging in production | Can't debug or audit | Add logging from day one
Overly broad regex | Blocks legitimate content | Use\bword boundaries
No alerts | Problems go unnoticed for days | Set a simple flag counter with threshold

---

## 3. Key Takeaways

- **Guardrails have two sides** — always filter both what goes *in* and what comes *out*.
- **The Moderation API** handles semantic harm (hate, violence). **Regex** handles structural patterns (phone numbers, banned words). Use both.
- **Logging is not optional** — it is your audit trail, your debugging tool, and your evidence if something goes wrong.
- **Alerts turn passive logs into active protection** — a threshold-based counter is enough to get started.
- Think of it as: **Moderate → Generate → Moderate → Log → Alert**. That five-step loop is your production safety net.

In the next topic, you will build on this by exploring **rate limiting and cost controls** — because guardrails protect content, but you also need to protect your API budget.

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