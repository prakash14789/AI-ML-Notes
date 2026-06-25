# 37. LLM Basics & Tokenisation - Suman - 12 Jan 2026

# LLM Basics & Tokenisation: Comprehensive Lecture Notes

In Class Resources:
[transformers_demo-main](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/f124bded-0f7b-412c-bd4e-7a4ba82f1995/01oxNJvzMUFxWaLJ.zip)

**Prerequisites:** Basic neural network concepts, Python programming, understanding of embeddings at a high level.

**Time to complete:** 40-50 minutes

**What you'll be able to do:**

- Explain the Transformer architecture's key innovations
- Implement and analyze BPE tokenisation
- Load and run inference on open-source 7B models
- Inspect token probabilities and understand model predictions

---

## 1. Introduction: What are LLMs and Why Should You Care?

### Core Definition

Large Language Models (LLMs) are neural networks trained on massive text corpora to predict the next token in a sequence. Built on the Transformer architecture, they capture statistical patterns in language, enabling them to generate coherent text, answer questions, translate languages, and perform reasoning tasks. The "large" refers to billions of parameters that store learned patterns.

### A Simple Analogy

Think of an LLM as an extremely sophisticated autocomplete system. Just as your phone predicts your next word while texting based on patterns in how people write, an LLM predicts the next token based on patterns learned from trillions of words of text. The difference is scale—LLMs have seen so much text that their predictions can seem intelligent.

This analogy breaks down because LLMs can generate complex multi-step reasoning and maintain context over thousands of words, which simple autocomplete cannot.

### Why This Matters to You

**Problem it solves:** Traditional NLP required hand-crafted features and separate models for each task. LLMs provide a general-purpose foundation that can be adapted to almost any text task through prompting or fine-tuning.

**What you'll gain:**

- Understanding of how ChatGPT, Claude, and other AI assistants work internally
- Ability to use open-source models for custom applications
- Knowledge to make informed decisions about model selection and deployment

**Real-world context:** GPT-4, Claude, LLaMA, Mistral—these models power applications used by hundreds of millions of people, from GitHub Copilot to customer service chatbots.

---

## 2. The Foundation: Core Concepts Explained

### Concept A: Transformer Architecture

**Definition:** The Transformer is a neural network architecture that processes sequences using self-attention mechanisms, allowing it to weigh the importance of different parts of the input when producing each output. Unlike RNNs that process tokens sequentially, Transformers process all tokens in parallel during training.

**Key characteristics:**

- **Self-attention:** Each token attends to all other tokens, capturing relationships regardless of distance
- **Parallel processing:** All positions compute simultaneously, enabling efficient GPU utilization
- **Position encoding:** Since there's no inherent order, position information is explicitly added

**A concrete example:**

```python
# Conceptual attention: for "The cat sat on the mat"
# When processing "sat", attention weights might be:
# "The": 0.1, "cat": 0.6, "sat": 0.1, "on": 0.1, "the": 0.05, "mat": 0.05
# High attention to "cat" because it's the subject of "sat"

```

**Common confusion:** Transformers don't "understand" language—they learn statistical correlations between tokens that often align with human understanding of meaning.

---

### Concept B: Tokenisation with BPE

**Definition:** Byte-Pair Encoding (BPE) is a subword tokenisation algorithm that starts with individual characters and iteratively merges the most frequent adjacent pairs to build a vocabulary. This creates a balance between character-level flexibility and word-level efficiency.

**How it relates to Transformers:** Tokenisation happens before the Transformer sees any input. The model only works with token IDs—it never sees raw text.

**Key characteristics:**

- **Subword splitting:** Rare words are split into familiar subwords ("unhappiness" → "un" + "happiness")
- **Fixed vocabulary:** Typically 32K-100K tokens for modern LLMs
- **Handling any input:** BPE can tokenise any text, even with misspellings or new words

**A concrete example:**

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Common word: single token
print(tokenizer.tokenize("hello"))  # ['hello']

# Rare word: multiple tokens
print(tokenizer.tokenize("cryptocurrency"))  # ['crypt', 'oc', 'urrency']

# Never-seen word: character-level fallback
print(tokenizer.tokenize("asdfghjkl"))  # ['asd', 'f', 'gh', 'j', 'kl']

```

**Remember:** Tokenisation significantly affects model behavior. A prompt that's 100 tokens in one tokenizer might be 150 in another, affecting context limits and costs.

---

### How Transformers and Tokenisation Work Together

Text enters as a string, gets split into tokens by BPE, converted to token IDs, embedded into vectors, processed through Transformer layers, and produces logits for the next token prediction. The tokenizer and model must match—using GPT-2's tokenizer with LLaMA would produce garbage.

---

## 3. Seeing It in Action: Worked Examples

**Tip:** Study these examples carefully before attempting the practice task. Understanding *why* each step is taken is more important than memorizing the steps.

### Example 1: Understanding BPE Tokenisation

**Scenario:** Explore how different text gets tokenised and understand the vocabulary structure.

**Our approach:** We'll tokenise various inputs and examine the token-to-text mapping.

**Step-by-step solution:**

```python
from transformers import AutoTokenizer

# Step 1: Load a tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(f"Vocabulary size: {len(tokenizer)}")  # 50257 tokens

# Step 2: Tokenize and decode various inputs
examples = [
    "Hello world",
    "The quick brown fox",
    "Machine learning is fascinating",
    "GPT-4 is amazing!",
    "你好世界",  # Chinese
]

for text in examples:
    tokens = tokenizer.encode(text)
    decoded = [tokenizer.decode([t]) for t in tokens]
    print(f"\nText: '{text}'")
    print(f"Token IDs: {tokens}")
    print(f"Tokens: {decoded}")

```

**Output:**

```
Vocabulary size: 50257

Text: 'Hello world'
Token IDs: [15496, 995]
Tokens: ['Hello', ' world']

Text: 'The quick brown fox'
Token IDs: [464, 2068, 7586, 21831]
Tokens: ['The', ' quick', ' brown', ' fox']

Text: 'Machine learning is fascinating'
Token IDs: [37573, 4673, 318, 21424]
Tokens: ['Machine', ' learning', ' is', ' fascinating']

Text: 'GPT-4 is amazing!'
Token IDs: [38, 11571, 12, 19, 318, 4998, 0]
Tokens: ['G', 'PT', '-', '4', ' is', ' amazing', '!']

```

**What just happened:** Common words become single tokens, while less common or compound terms get split. Notice how "GPT-4" becomes 4 tokens because it's a newer term not in GPT-2's vocabulary as a unit.

**Check your understanding:** Why does "Hello" have no space but " world" has a leading space in the tokens?

---

### Example 2: Loading a 7B Model for Inference

**Scenario:** You want to run a local open-source LLM to generate text.

**What's different:** We load a full model and generate text autoregressively.

**Solution:**

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Step 1: Load model and tokenizer (using smaller model for demo)
model_name = "gpt2"  # Replace with "meta-llama/Llama-2-7b-hf" for 7B
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# For 7B models, load in 8-bit to fit in memory:
# model = AutoModelForCausalLM.from_pretrained(
#     "meta-llama/Llama-2-7b-hf",
#     load_in_8bit=True,
#     device_map="auto"
# )

# Step 2: Prepare input
prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

# Step 3: Generate
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

# Step 4: Decode and print
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)

```

**Output:**

```
The future of artificial intelligence is both exciting and uncertain.
As AI systems become more capable, we must carefully consider the
implications for society, employment, and human autonomy...

```

**Key lesson:** The same pipeline works for any causal LM—change the model name to switch from GPT-2 to LLaMA to Mistral.

---

### Example 3: Inspecting Logits and Token Probabilities

**Background:** Understanding what the model actually predicts helps with debugging and building applications.

**The challenge:** Examine the probability distribution over next tokens before sampling.

**The approach:** Extract logits, apply softmax, and analyze top predictions.

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Input prompt
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt")

# Get model outputs (logits)
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits[0, -1, :]  # Shape: [vocab_size]

# Convert logits to probabilities
probs = F.softmax(logits, dim=-1)

# Get top 10 predictions
top_k = 10
top_probs, top_indices = torch.topk(probs, top_k)

print(f"Prompt: '{prompt}'\n")
print("Top 10 next-token predictions:")
print("-" * 40)
for prob, idx in zip(top_probs, top_indices):
    token = tokenizer.decode([idx])
    print(f"'{token}': {prob.item()*100:.2f}%")

```

**Output:**

```
Prompt: 'The capital of France is'

Top 10 next-token predictions:
----------------------------------------
' Paris': 89.34%
' the': 2.15%
' a': 1.02%
' located': 0.87%
' known': 0.45%
' called': 0.32%
' in': 0.28%
' Paris': 0.21%
' one': 0.18%
' not': 0.15%

```

**Why this approach:** Logit inspection reveals model confidence and alternatives. Here, the model is 89% confident about "Paris"—a well-calibrated prediction for a factual question.

**Caution:** Low probability doesn't mean wrong. Creative writing benefits from sampling lower-probability tokens; factual tasks should use high-probability ones.

---

## 4. Common Pitfalls: What Can Go Wrong and How to Avoid It

- **The Mistake:** Using mismatched tokenizer and model

**Why It's a Problem:** Token IDs mean completely different things across tokenizers; output will be garbage
**The Right Approach:** Always load tokenizer and model from the same checkpoint

`# CORRECT
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")`

**Why This Works:** Each model is trained with a specific tokenizer; they must match

---

- **The Mistake:** Not accounting for tokenisation when designing prompts

**Why It's a Problem:** What looks like one word might be 5 tokens, affecting context limits and costs
**The Right Approach:** Check token counts for important prompts

`tokens = tokenizer.encode(prompt)
print(f"Prompt uses {len(tokens)} tokens")`

**Why This Works:** Token-awareness helps optimize prompt engineering and cost management

---

- **The Mistake:** Assuming generation is deterministic

**Why It's a Problem:** With sampling enabled (temperature > 0), outputs vary each run
**The Right Approach:** Set seed for reproducibility, or use temperature=0 for deterministic output

`torch.manual_seed(42)
outputs = model.generate(..., do_sample=True, temperature=0.7)
# OR
outputs = model.generate(..., do_sample=False)  # Greedy, deterministic`

**Why This Works:** Controlled randomness or no randomness enables reproducible results

**If you're stuck:** Start with the smallest model (GPT-2) to verify your pipeline works before scaling to 7B+ models.

---

## 5. Your Turn: Practice & Self-Assessment

### Practice Task (Estimated: 25 minutes)

**The Challenge:** Build a token analysis tool that provides insights about how text gets processed by an LLM.

**Specifications:**

- Load any HuggingFace causal LM tokenizer
- Create a function that takes text and returns: token count, tokens as strings, token IDs
- Analyze the top-10 predicted next tokens for a given prompt
- Compare tokenisation across two different models (e.g., GPT-2 vs. Llama-2)

**Hint:** Structure your solution as reusable functions: `analyze_tokens(text, tokenizer)` returns a dict with token info, `get_next_token_predictions(prompt, model, tokenizer, top_k)` returns ranked predictions. For model comparison, focus on how the same text produces different token counts.

**Extension (optional):** Add visualization showing how token probabilities change with different temperatures.

---

### Check Your Understanding

1. 
**Explanation question:** Why do LLMs use subword tokenisation instead of word-level tokenisation? Explain using an example with a rare word.

2. 
**Application question:** You're building a chatbot with a 4096 token context limit. The user's message uses 500 tokens and you want to include conversation history. How would you manage this constraint?

3. 
**Error analysis:**

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
outputs = model(tokenizer("Hello", return_tensors="pt")["input_ids"])

```

What's wrong with this code, and what would the output be?

1. **Transfer question:** How would you modify logit inspection to implement "classifier-free guidance" where you want to avoid generating certain tokens?

**Answers & Explanations:**

1. 
Word-level tokenisation would require an infinitely large vocabulary to handle all possible words, including typos, new coinages, and compound words. Subword tokenisation like BPE can represent "cryptocurrency" as "crypt" + "oc" + "urrency"—all subwords likely seen in training—even if "cryptocurrency" wasn't common when the tokenizer was built.

2. 
Use a sliding window or summarization: keep recent messages (most relevant) within budget, summarize older context, or truncate from the start. Track running token count and implement a maximum conversation history policy.

3. 
Tokenizer/model mismatch. GPT-2 tokenizer produces token IDs that mean different things in LLaMA's vocabulary. The output would be nonsensical because the model interprets GPT-2's token 15496 ("Hello") as a completely different token in LLaMA's vocabulary.

4. 
Before sampling, set the logits for unwanted tokens to a very large negative number (e.g., -inf), which makes their probability after softmax essentially zero. This is called "logit biasing" or "token banning."

---

### Self-Assessment Checklist

You've mastered this topic if you can:

- Explain how self-attention enables Transformers to capture long-range dependencies
- Implement BPE tokenisation analysis for any text
- Load and run inference on open-source LLMs
- Inspect and interpret token probabilities from model outputs
- Understand the relationship between temperature and output diversity
- Debug tokenizer/model mismatches and other common issues

**If you checked fewer than 5 boxes:** Run the examples with different models and prompts to build intuition.

---

## 6. Consolidation: Key Takeaways & Next Steps

### The Essential Ideas

- **Transformers process in parallel:** Self-attention allows seeing all tokens at once, unlike sequential RNNs
- **Tokenisation is crucial:** BPE balances vocabulary size with ability to represent any text
- **Generation is autoregressive:** Each new token depends on all previous tokens

### Mental Model Check

By now, you should think of LLMs as: Statistical pattern-completion engines that convert text to tokens, process through attention layers to capture relationships, and output probability distributions over next tokens that can be sampled for generation.

### What You Can Now Do

You can load open-source LLMs, understand their tokenisation, analyze their predictions, and use them for text generation. This foundation enables fine-tuning, prompt engineering, and building AI applications.

### Next Steps

**To deepen this knowledge:** Experiment with different sampling strategies (temperature, top-p, top-k) and observe how they affect output quality.

**To build on this:** Learn about fine-tuning (LoRA, full fine-tuning) and advanced prompting techniques.

**Additional resources:** HuggingFace Transformers documentation, "Attention Is All You Need" paper, LLaMA and Mistral model cards.

---

## Quick Reference Card

Component | Purpose | Example Code
Tokenizer | Text → Token IDs | tokenizer.encode("hello")
Embedding | Token IDs → Vectors | model.transformer.wte(input_ids)
Forward pass | Vectors → Logits | model(input_ids).logits
Softmax | Logits → Probabilities | F.softmax(logits, dim=-1)
Sampling | Probabilities → Token | torch.multinomial(probs, 1)
Decode | Token ID → Text | tokenizer.decode([token_id])

---

**Questions or stuck?** HuggingFace documentation and the Transformers GitHub discussions are excellent resources for troubleshooting.

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