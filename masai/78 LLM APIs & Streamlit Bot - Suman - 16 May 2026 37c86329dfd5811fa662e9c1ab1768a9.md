# 78. LLM APIs & Streamlit Bot - Suman - 16 May 2026

# LLM APIs and Streamlit Bot

## 1. What You'll Learn in This Section

In this lesson, you'll learn to:

- Explain how Large Language Models work and distinguish between cloud-based, local, open, and closed models
- Compare fine-tuning techniques — full fine-tuning, transfer learning, LoRA, and QLoRA — and identify when each applies
- Call the OpenAI and Anthropic APIs from Python, using roles, temperature, max tokens, streaming, and f-string prompt templates
- Build a working web chatbot by combining Streamlit with an LLM API

## 2. Detailed Explanation

### What is a Large Language Model

A **Large Language Model (LLM)** is an AI model trained on massive amounts of text that generates human-like language one token at a time — like a very well-read autocomplete engine.

**Why it matters**

LLMs are the engine behind products you interact with every day, from customer support bots to code assistants. Understanding what they are makes every other concept in this topic click into place.

**Walkthrough**

The key mechanism is next-token prediction. Given "the sky is", the model predicts "blue" as the most likely next word. It repeats this process token by token to produce answers, code, summaries, translations, or solutions.

Well-known LLMs include:

Model | Company
ChatGPT (GPT family) | OpenAI
Claude | Anthropic
Gemini | Google
Llama | Meta

**Common mistakes**

- Thinking an LLM "understands" text the way humans do — it predicts tokens based on patterns, not meaning.
- Treating a single model name as fixed; providers release new versions regularly, so always check the model identifier you pass in code.

---

### Cloud-based vs local LLMs

**Cloud-based (API) LLMs** run on the provider's server. Your app sends a prompt over the internet and receives a response. **Local LLMs** run entirely on your own machine — you download the model and run it without an internet connection.

**Why it matters**

Choosing the wrong deployment model can mean paying unnecessary API fees, exposing sensitive data, or building an app that runs too slowly for your users.

**Walkthrough**

Here is how each approach works:

**Cloud-based flow:**

1. A user writes a prompt in your application.
2. Your application sends the prompt to the provider's server via API.
3. The server generates a response.
4. Your application receives and displays the response.

**Local flow:**

1. A user sends a prompt to your local application.
2. The local machine processes the request.
3. The model generates a response locally.
4. The response is displayed.

The decision table below captures the key trade-offs:

Dimension | Cloud-based (API) | Local
Privacy | Data leaves the device | Data stays on device
Internet required | Yes | No
Speed | Generally faster | Generally slower
Output quality | Higher (full-size models) | Lower (smaller/quantized)
Hardware needed | None | Strong GPU/CPU
Setup | Just call the API | Install tool + download model
Cost model | API fees | No per-call fees; hardware cost upfront

**Choose API-based LLMs when** rapid development is the priority, you need the most advanced models, you are building SaaS apps or web chatbots, or your local hardware is insufficient.

**Choose local LLMs when** privacy is critical (defense, healthcare, legal), offline operation is required, you are fine-tuning a model for a specific domain, or budget constraints make recurring API fees undesirable.

The distinction maps to a familiar analogy: API-based is like using Google Docs — the editor runs on the cloud. Local is like running Microsoft Word — everything runs on your machine.

**Common mistakes**

- Assuming local is always more private — if you use a cloud-hosted open model, data still leaves your device.
- Underestimating local hardware requirements; running a capable LLM demands a strong GPU/CPU.

---

### Why local LLMs need specialized tools

When you download a local LLM, what arrives is not a ready-to-run program — it is a set of raw files that need software to execute.

**Why it matters**

Downloading a model without understanding this step leads to confusion about why nothing "just runs" after the download finishes.

**Walkthrough**

A downloaded model package typically contains:

- Model weight files
- Tokenizer files
- Configuration files

These files cannot execute by themselves. A supporting tool must load the weights into RAM, perform tensor computations, manage GPU and CPU resources, and handle tokenization and next-token generation.

Think of it like downloading a movie file without a media player. The file exists on disk but cannot play without software such as VLC. Downloaded LLM files need a tool for the same reason.

Popular tools:

Tool | Notes
Ollama | Widely used; now also available via cloud API
LM Studio | Desktop interface for local models
llama.cpp | Low-level C++ runtime for quantized models

The typical workflow: install the tool → download the desired model → use the tool to load and run it.

**Common mistakes**

- Trying to run the weight files directly — they are data, not executables.
- Skipping environment configuration; each tool has its own setup steps.

---

### Open vs closed LLMs

**Open LLMs** make their model weights publicly available — you can download, fine-tune, and modify them. **Closed LLMs** keep their weights private — you can only use them through an API.

**Why it matters**

"Open vs closed" and "local vs cloud" are two independent axes. Understanding both helps you accurately describe any model and know what you can do with it.

**Walkthrough**

```
Any LLMWhere does it run?Are weights public?Cloud — runs on provider serverLocal — runs on your machineOpen — weights downloadable,fine-tuning allowedClosed — weights private,API-only access
```

Examples across both axes:

Model / scenario | Local? | Open?
Llama running on a laptop | Yes | Yes
ChatGPT | No | No
Open model hosted on a cloud server | No | Yes

Note that an open model can run either locally or in the cloud. The person who deploys it is responsible for infrastructure, hardware, updates, and maintenance.

**Common mistakes**

- Conflating "open" with "local" — an open model can be hosted on a cloud server.
- Assuming open means fully transparent; the degree of openness varies (some release only weights, others also release code).

---

### Fine-tuning techniques for LLMs

**Fine-tuning** is the process of adapting a model's weights to perform better on a specific task or domain. Four techniques exist, ranging from computationally expensive to highly efficient.

**Why it matters**

Out-of-the-box LLMs are generalists. Fine-tuning lets you create a specialist — a model that excels at medical documentation, legal analysis, or your company's specific use case.

**Walkthrough**

**Full fine-tuning:** All model parameters are retrained from scratch. Weights start at random values and are updated iteratively using training data. This is computationally expensive and requires large amounts of data — seldom practical for individual developers.

**Transfer learning:** Rather than starting from random weights, you start from a pre-trained model whose weights were learned on a large dataset. Those weights are transferred to the new task and refined with task-specific data. This is far less expensive than full fine-tuning.

Example: a neural network trained to classify dogs vs cats already understands low-level features (edges, textures, intensities). Those weights become the starting point for a new task such as classifying cars vs bicycles. Well-known pre-trained CNN models include EfficientNet, VGG16, and ResNet.

You can also swap out layers. If a pre-trained model has a 20-class output layer but you only need 10 classes, replace and retrain just the final layer while keeping earlier layers fixed.

**LoRA (low-rank adaptation):** A parameter-efficient fine-tuning technique that inserts small adapter layers rather than updating all model weights. The result is less robust than full fine-tuning but far cheaper to compute.

**QLoRA (quantized LoRA):** The same adapter-based approach as LoRA, but with quantized weights. Quantization drastically reduces memory requirements, making it practical to fine-tune and run LLMs on edge devices or machines with limited memory.

Technique | Starting weights | Cost | Memory
Full fine-tuning | Random | Very high | Very high
Transfer learning | Pre-trained | Medium | Medium
LoRA | Pre-trained | Low | Medium
QLoRA | Pre-trained (quantized) | Low | Very low

**Common mistakes**

- Confusing "pre-trained" with "fine-tuned" — pre-trained means the weights are already meaningful from large-scale training, not that they are specialized for your task yet.
- Treating LoRA and QLoRA as identical — QLoRA adds quantization on top of LoRA, which cuts memory requirements dramatically.

---

### Setting up an LLM API environment

Before making any API calls, you need to install the right libraries, create an API key, and store it securely.

**Why it matters**

Skipping any of these steps — especially secure key storage — leads to broken code or, worse, leaked credentials that generate unexpected charges on your account.

**Walkthrough**

Install all required libraries in one command:

```bash
pip install openai anthropic streamlit python-dotenv

```

Then follow these four steps:

1. **Install libraries** for your chosen provider (OpenAI, Anthropic) and supporting tools.
2. **Create an API key** on the provider's website (e.g., the OpenAI developer portal). The key uniquely identifies your account, authenticates requests, and enables the provider to track usage and billing.
3. **Store the API key securely** using a `.env` file and the `python-dotenv` library. The `.env` file holds secret configuration variables that should never be hardcoded in source code. Call `load_dotenv()` at the top of your script to read the file and make its values available as environment variables.
4. **Write the API call code** and retrieve the response.

**Common mistakes**

- Hardcoding the API key directly in your Python file — if you push that file to GitHub, your key is exposed to anyone who views the repo.
- Forgetting to call `load_dotenv()` before accessing `os.getenv()` — without it, the `.env` file is never read.

---

### OpenAI API call structure

The OpenAI API follows a consistent three-step pattern: create a client, send a request with a messages list, and read the response.

**Why it matters**

Once you know this pattern for OpenAI, you can apply nearly the same structure to Anthropic and other providers — the differences are mostly naming.

**Walkthrough**

```python
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Step 1: create a client object using the API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Step 2: send a request to the model
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "system", "content": "You are a helpful teacher."},
        {"role": "user",   "content": "What is AI?"}
    ]
)

# Step 3: print the response
print(response.choices[0].message.content)

```

What each part does:

- `OpenAI(api_key=...)` — creates a client object that establishes the authenticated connection to OpenAI's servers.
- `client.chat.completions.create(...)` — the main method that sends the prompt to the model and returns a response.
- `model` — specifies which OpenAI model to use (e.g., `gpt-4.1-mini`).
- `messages` — the list of message objects that form the prompt.

**Common mistakes**

- Passing the API key as a plain string instead of reading it from an environment variable.
- Accessing the response content with the wrong path — it is `response.choices[0].message.content`, not `response.content`.

---

### The messages structure and roles

Every API call uses a `messages` list. Each item has two fields: **`role`** (who is sending the message) and **`content`** (the actual text).

**Why it matters**

The role tells the model how to interpret each message. Using the wrong role — or skipping the system role — leads to a bot that ignores your instructions or lacks a consistent persona.

**Walkthrough**

There are three roles:

**System role** — sets the behavior or persona of the LLM at the start of a conversation. Used once, at the beginning.

```python
{"role": "system", "content": "You are a helpful teacher."}

```

**User role** — represents the human's question or prompt.

```python
{"role": "user", "content": "What is AI?"}

```

**Assistant role** — passes the model's previous response back to it. This is how conversation history is maintained.

```python
{"role": "assistant", "content": "AI stands for Artificial Intelligence."}

```

**Common mistakes**

- Placing the system message after user messages — it must come first for the model to apply it correctly.
- Redefining the system role in every turn — set it once and keep it fixed.

---

### Stateless LLMs and conversation history

Most LLMs are **stateless** — they do not remember previous interactions. Every new API call is treated as if no prior conversation has occurred.

**Why it matters**

If you do not manually replay the conversation history, your chatbot will "forget" what was said. This is one of the most common sources of broken multi-turn conversations.

**Walkthrough**

To simulate a conversation, include all previous messages in each new API call. The assistant role lets you replay the model's earlier response so the model understands the context.

First call:

```python
messages=[
    {"role": "system",    "content": "You are a helpful teacher."},
    {"role": "user",      "content": "What is AI?"}
]
# Model responds: "AI stands for Artificial Intelligence."

```

Second call — replay the full conversation including the system message and the previous response as an assistant message:

```python
messages=[
    {"role": "system",    "content": "You are a helpful teacher."},
    {"role": "user",      "content": "What is AI?"},
    {"role": "assistant", "content": "AI stands for Artificial Intelligence."},
    {"role": "user",      "content": "Give one example."}
]

```

The model uses the assistant message to understand what it already said and generates an appropriate follow-up.

**Common mistakes**

- Sending only the latest user message — the model has no context and produces disconnected replies.
- Letting the messages list grow indefinitely — very long histories can exceed the model's context window and cause errors.

---

### Temperature and max tokens

**Temperature** controls the creativity and randomness of a response. **Max tokens** caps the maximum length of the generated output.

**Why it matters**

Without tuning these parameters, your bot may give robotic one-word answers or, conversely, ramble endlessly. Matching the right values to your use case dramatically improves output quality.

**Walkthrough**

**Temperature:**

- Low temperature (close to 0): responses are predictable and focused. Asked "What is the color of the sky?", the model answers "Blue."
- High temperature: responses are more creative and varied. The model might note that the sky is normally blue but can appear red at sunset.

**Max tokens:** limits the maximum length of the generated response. Useful for controlling output size and handling response data more predictably.

```python
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[...],
    temperature=0.7,
    max_tokens=256
)

```

**Common mistakes**

- Setting temperature to 0 for a creative writing bot — you get repetitive, boring output.
- Forgetting to set `max_tokens` in production — unexpectedly long responses increase latency and API costs.

---

### Python f-strings for prompt templates

Rather than hardcoding a prompt, a common pattern uses **Python f-strings** with variables — letting you build reusable, parameterized prompts.

**Why it matters**

Hardcoded prompts mean rewriting code every time the topic changes. F-strings turn your prompt into a template that works for any input.

**Walkthrough**

```python
user_topic = "machine learning"
prompt = f"Explain {user_topic} in simple terms."
# Result: "Explain machine learning in simple terms."

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": prompt}]
)

```

The value of `user_topic` can be set dynamically — for example, read from user input — making the same template reusable for different topics.

**Common mistakes**

- Using string concatenation (`"Explain " + user_topic + " in simple terms."`) instead of f-strings — it works but is harder to read and maintain.
- Forgetting the `f` prefix before the string — without it, `{user_topic}` is printed literally instead of being replaced.

---

### Anthropic API call structure

The Anthropic API follows the same logical pattern as OpenAI's, with different constructor and method names.

**Why it matters**

Once you see that both providers share the same structure — client creation, messages list, response extraction — switching between them feels natural rather than starting from scratch.

**Walkthrough**

```python
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()

# Step 1: create a client using the Anthropic constructor
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Step 2: send a request to the Claude model
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "What is AI?"}]
)

# Step 3: print the response
print(response.content[0].text)

```

Key differences from OpenAI:

Aspect | OpenAI | Anthropic
Client constructor | OpenAI(api_key=...) | anthropic.Anthropic(api_key=...)
Request method | client.chat.completions.create(...) | client.messages.create(...)
Response content | response.choices[0].message.content | response.content[0].text
Role/content structure | Same | Same

Providers intentionally keep structural patterns similar so developers can migrate between APIs without relearning everything.

**Common mistakes**

- Using `response.choices[0].message.content` with the Anthropic client — Anthropic uses `response.content[0].text`.
- Forgetting that Anthropic requires `max_tokens` to be set explicitly, while OpenAI makes it optional.

---

### Streaming responses

**Streaming** is a response-delivery mode where the LLM sends tokens one at a time as they are generated, rather than waiting to compile the complete response first.

**Why it matters**

Streaming makes your chatbot feel faster and more interactive. Users see the first words of a response almost immediately rather than staring at a blank screen while the model finishes generating.

**Walkthrough**

Enable streaming by setting `stream=True`. Tokens arrive in a loop and are printed as they come in.

```python
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "user", "content": "Explain neural networks."}],
    temperature=0.7,
    stream=True
)

for chunk in response:
    token = chunk.choices[0].delta.content
    if token is not None:
        print(token, end="", flush=True)

```

Benefits of streaming:

- Faster perceived response time — the user sees output immediately.
- Real-time token generation — output appears gradually as it is produced.
- More natural and interactive conversation feel.

One important point: "streaming" (the token-delivery mechanism) and "Streamlit" (the Python web framework) are unrelated concepts despite their similar names.

**Common mistakes**

- Trying to access `response.choices[0].message.content` when streaming — in streaming mode, the response object is an iterator of chunks, not a single object.
- Forgetting `flush=True` in `print()` — without it, output may buffer and appear in bursts rather than token by token.

---

### Streamlit — building web interfaces for AI applications

**Streamlit** is a Python framework for building interactive web applications quickly, with particular utility for AI and machine learning projects. It removes the need to write HTML, CSS, or JavaScript.

**Why it matters**

Without Streamlit, a Python developer who wants a web interface must learn front-end development. Streamlit collapses that barrier to a handful of Python function calls.

**Walkthrough**

A minimal Streamlit app looks like this:

```python
import streamlit as st

st.title("My First App")
name = st.text_input("Enter your name")

if name:
    st.write(f"Hello, {name}!")

```

This creates a web page with a title and a text input box. Whatever the user types is captured and displayed immediately.

Key UI components for chatbot applications:

- `st.title()` — sets the page title.
- `st.text_input()` — creates a text input box to capture user prompts.
- `st.write()` / `st.markdown()` — displays text or formatted output.

Advantages of Streamlit:

- Fast development: interactive web apps with pure Python, no front-end expertise needed.
- Interactive UI: widgets like text inputs, sliders, and buttons created with single function calls.
- AI-friendly: integrates naturally with LLM API calls.
- Cloud deployment support: Streamlit Community Cloud lets you deploy apps directly.

**Common mistakes**

- Confusing Streamlit with streaming — Streamlit is a web UI framework; streaming is a response-delivery mode. They are independent and can be used together or separately.
- Forgetting to run `streamlit run app.py` instead of `python app.py` — Streamlit apps must be launched with the Streamlit CLI.

---

### Building a Streamlit bot

A **Streamlit bot** is a web chatbot that combines Streamlit (the UI layer) with an LLM API (the AI response layer). The app captures user input, sends it to the LLM, and displays the response in the browser.

**Why it matters**

This combination lets you turn a few dozen lines of Python into a shareable, interactive AI application — no server configuration, no front-end code, no infrastructure to manage.

**Walkthrough**

The complete flow through a Streamlit + LLM application:

```
User types a promptStreamlit UIcaptures prompt via st.text_inputAPI call to OpenAI or Anthropicwith role, content, model, temperatureLLM on provider servergenerates response tokensAPI response returnedstreaming or bulkStreamlit app displaysresponse to user
```

A working Streamlit bot:

```python
import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.title("AI Chatbot")
user_input = st.text_input("Ask a question:")

if user_input:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": user_input}],
        stream=True
    )
    for chunk in response:
        token = chunk.choices[0].delta.content
        if token is not None:
            st.write(token)

```

The pattern to notice: wherever `st.*` calls appear, that is Streamlit managing the UI. Everywhere else, the standard LLM API call pattern applies.

Real-world applications built this way include: customer support chatbots, personal assistant systems, medical chatbots, educational tutors, code-generation assistants, and business analytics assistants.

**Common mistakes**

- Placing the `client = OpenAI(...)` call inside the user-input block — the client gets re-created on every interaction. Move it outside, before the `st.*` calls.
- Not handling the case where `user_input` is empty — always gate your API call with an `if user_input:` check.

---

## 3. Key Takeaways

- LLMs generate text **token by token**. Cloud-based API vs local deployment involves trade-offs across privacy, speed, output quality, and cost — use the comparison table to guide that decision.
- The **messages list** is the heart of every API call. Three roles — system (persona), user (question), assistant (previous reply) — control behavior and maintain conversation history despite the model being stateless.
- **Temperature** and **max tokens** are your primary levers for shaping response quality: temperature controls creativity, max tokens controls length.
- Both OpenAI and Anthropic follow the same three-step pattern: create a client, call the request method with a messages list, extract the response. The differences are naming only.
- **Streamlit** turns LLM API code into a shareable web application with almost no extra effort. Pair `st.text_input()` with any LLM API call and you have a working chatbot.

**Mental model:** Building an LLM-powered web app is like connecting three pipes. A Streamlit UI pipe catches user input. An API pipe sends that input to a server-side model. A response pipe streams tokens back to the screen. Your job as the developer is to connect those three pipes correctly.

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