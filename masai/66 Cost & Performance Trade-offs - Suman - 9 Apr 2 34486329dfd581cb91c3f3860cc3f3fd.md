# 66. Cost & Performance Trade-offs - Suman - 9 Apr 2026

# Lecture Notes: Cost & Performance Trade-offs in LLM Applications

### PPT File: [Click Here](https://drive.google.com/file/d/1m1GpEVbRvrfzzaL4AO1XjVY4w6jQyRQx/view?usp=sharing)

### Colab File: [Click Here](https://drive.google.com/file/d/1bGB2gEFZxdGwcY6FkYfBVZm_CleaBW0U/view?usp=sharing)

## Session Overview

This session covers the critical economic and performance considerations when deploying Large Language Models in production. You'll learn how to optimize token usage, manage context windows, maximize GPU utilization, implement effective batching, and make informed cost-performance tradeoffs.

**Duration:** 3 hours

**Prerequisites:** Understanding of LLMs, basic Python, API usage

## Learning Objectives

By the end of this session, you will be able to:

1. **Calculate LLM costs** accurately using token-based pricing
2. **Optimize context windows** to minimize token usage
3. **Maximize GPU efficiency** through batching strategies
4. **Implement cost-saving techniques** like caching and model routing
5. **Make informed tradeoffs** between cost, latency, and quality
6. **Monitor and optimize** production LLM applications

---

## Part 1: Token Economics

### Understanding Token-Based Pricing

**Core Concept:** LLM APIs charge per token, not per API call or word.

**Token Definition:**

```
Token ≠ Word

Tokens are subword units created by tokenization algorithms (BPE)

Examples:
"Hello" → ["Hello"] (1 token)
"Hello world" → ["Hello", " world"] (2 tokens)
"supercalifragilisticexpialidocious" → 
  ["super", "cal", "if", "rag", "il", "istic", "exp", "ial", "id", "oci", "ous"]
  (11 tokens)

Average ratio: 1 word ≈ 1.3 tokens (English)

```

### Pricing Models (OpenAI GPT-4 Example)

```
Model: GPT-4 (8K context)
Input pricing: $0.03 per 1,000 tokens
Output pricing: $0.06 per 1,000 tokens

Model: GPT-3.5-turbo
Input pricing: $0.001 per 1,000 tokens
Output pricing: $0.002 per 1,000 tokens

GPT-4 is 30× more expensive than GPT-3.5!

```

### Calculating Request Costs

**Formula:**

```
Cost per request = (Input tokens × Input price per 1K / 1000) 
                 + (Output tokens × Output price per 1K / 1000)

```

**Example 1: Simple Query**

```python
import tiktoken

# Initialize tokenizer for GPT-4
enc = tiktoken.encoding_for_model("gpt-4")

# Input
user_query = "What is the capital of France?"
system_prompt = "You are a helpful assistant."

input_text = system_prompt + "\n" + user_query
input_tokens = len(enc.encode(input_text))

# Output
response = "The capital of France is Paris."
output_tokens = len(enc.encode(response))

# Calculate cost
input_cost = (input_tokens / 1000) * 0.03
output_cost = (output_tokens / 1000) * 0.06
total_cost = input_cost + output_cost

print(f"Input tokens: {input_tokens}")
print(f"Output tokens: {output_tokens}")
print(f"Total cost: ${total_cost:.6f}")

# Output:
# Input tokens: 13
# Output tokens: 8
# Total cost: $0.000870

```

**Example 2: Long Document Analysis**

```python
# Long document
document = """[10,000 word article about AI]"""
system_prompt = "Summarize this article in 3 paragraphs."

input_text = system_prompt + "\n" + document
input_tokens = len(enc.encode(input_text))  # ~13,000 tokens

# Expected output
output_tokens = 200  # Estimated summary length

# Calculate cost
input_cost = (input_tokens / 1000) * 0.03   # $0.39
output_cost = (output_tokens / 1000) * 0.06  # $0.012
total_cost = input_cost + output_cost        # $0.402

print(f"Cost for this request: ${total_cost:.3f}")

# At 1,000 requests/day:
daily_cost = total_cost * 1000  # $402/day
monthly_cost = daily_cost * 30  # $12,060/month

print(f"Monthly cost at scale: ${monthly_cost:,.0f}")

```

### Token Counting Tools

```python
import tiktoken

def count_tokens(text, model="gpt-4"):
    """Count tokens in text for a specific model."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def estimate_cost(input_text, output_tokens, model="gpt-4"):
    """Estimate cost for a request."""
    pricing = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.001, "output": 0.002},
    }
    
    input_tokens = count_tokens(input_text, model)
    
    input_cost = (input_tokens / 1000) * pricing[model]["input"]
    output_cost = (output_tokens / 1000) * pricing[model]["output"]
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost
    }

# Example usage
text = "Your input text here..." * 1000
cost = estimate_cost(text, output_tokens=500)

print(f"Total cost: ${cost['total_cost']:.4f}")

```

### Token Optimization Strategies

**1. Truncate Unnecessary Context**

```python
def truncate_to_token_limit(text, max_tokens=3000, model="gpt-4"):
    """Truncate text to fit within token limit."""
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    # Truncate and decode
    truncated_tokens = tokens[:max_tokens]
    truncated_text = enc.decode(truncated_tokens)
    
    return truncated_text

# Example
long_text = "..." * 100000
optimized = truncate_to_token_limit(long_text, max_tokens=2000)

print(f"Original: {count_tokens(long_text)} tokens")
print(f"Optimized: {count_tokens(optimized)} tokens")
print(f"Savings: {count_tokens(long_text) - count_tokens(optimized)} tokens")

```

**2. Remove Redundant Information**

```python
def remove_duplicates(conversation_history):
    """Remove duplicate information from history."""
    seen = set()
    unique_messages = []
    
    for message in conversation_history:
        # Create hash of message content
        content_hash = hash(message["content"])
        
        if content_hash not in seen:
            seen.add(content_hash)
            unique_messages.append(message)
    
    return unique_messages

# Example
history = [
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "AI is..."},
    {"role": "user", "content": "What is AI?"},  # Duplicate!
    {"role": "user", "content": "Tell me more"}
]

optimized_history = remove_duplicates(history)
print(f"Reduced from {len(history)} to {len(optimized_history)} messages")

```

**3. Compress System Prompts**

```python
# Inefficient system prompt (verbose)
verbose_prompt = """
You are a helpful AI assistant. You should always be polite and 
courteous. You should provide accurate information. You should 
ask clarifying questions when needed. You should format your 
responses clearly. You should cite sources when possible.
"""

# Optimized system prompt (concise)
concise_prompt = """
You are a helpful AI assistant. Be polite, accurate, and clear. 
Ask clarifying questions and cite sources.
"""

print(f"Verbose: {count_tokens(verbose_prompt)} tokens")
print(f"Concise: {count_tokens(concise_prompt)} tokens")
print(f"Savings: {count_tokens(verbose_prompt) - count_tokens(concise_prompt)} tokens")

# At 1 million requests:
savings_per_million = (count_tokens(verbose_prompt) - count_tokens(concise_prompt)) * 1_000_000
cost_savings = (savings_per_million / 1000) * 0.03
print(f"Cost savings: ${cost_savings:.2f} per million requests")

```

---

## Part 2: Context Length Management

### Context Window Limits

**Definition:** Maximum number of tokens (input + output) a model can process in one request.

```
Model Context Windows:

GPT-3.5-turbo: 4,096 tokens
GPT-4: 8,192 tokens
GPT-4-32k: 32,768 tokens
GPT-4-turbo: 128,000 tokens
Claude-2: 100,000 tokens
Claude-3: 200,000 tokens

Context = Input tokens + Output tokens

Example (GPT-4):
Max context: 8,192 tokens
Input: 7,000 tokens
Available for output: 1,192 tokens

```

### The Context Window Problem

**Scenario:** Chatbot with conversation history

```python
class Chatbot:
    def __init__(self):
        self.history = []
        self.max_context = 4096  # GPT-3.5 limit
    
    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})
    
    def get_total_tokens(self):
        full_history = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in self.history
        ])
        return count_tokens(full_history)
    
    def chat(self, user_message):
        self.add_message("user", user_message)
        
        # Check context limit
        total_tokens = self.get_total_tokens()
        
        if total_tokens > self.max_context - 500:  # Reserve 500 for response
            print(f"WARNING: Context full ({total_tokens} tokens)")
            return None
        
        # Make API call...
        return response

# Problem: After 20 exchanges, context fills up!
bot = Chatbot()
for i in range(20):
    bot.chat(f"Question {i+1}")
    # Eventually: WARNING: Context full

```

### Context Management Strategies

**1. Sliding Window (Keep Recent History)**

```python
class SlidingWindowChatbot:
    def __init__(self, max_messages=10):
        self.history = []
        self.max_messages = max_messages
        self.system_prompt = "You are a helpful assistant."
    
    def add_message(self, role, content):
        self.history.append({"role": role, "content": content})
        
        # Keep only last N messages
        if len(self.history) > self.max_messages:
            # Keep system prompt + recent messages
            self.history = self.history[-self.max_messages:]
    
    def get_context(self):
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        return messages

# Example
bot = SlidingWindowChatbot(max_messages=6)

for i in range(20):
    bot.add_message("user", f"Question {i+1}")
    bot.add_message("assistant", f"Answer {i+1}")
    print(f"History length: {len(bot.history)}")

# Output: History never exceeds 6 messages

```

**2. Summarization (Compress Old History)**

```python
class SummarizingChatbot:
    def __init__(self):
        self.history = []
        self.summary = ""
        self.max_history_tokens = 2000
    
    def should_summarize(self):
        history_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in self.history
        ])
        return count_tokens(history_text) > self.max_history_tokens
    
    def summarize_history(self):
        """Compress old history into summary."""
        history_text = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in self.history[:-3]  # Keep last 3 exchanges
        ])
        
        # Call LLM to summarize
        summary_prompt = f"Summarize this conversation:\n{history_text}"
        summary = call_llm(summary_prompt, max_tokens=200)
        
        # Update state
        self.summary = summary
        self.history = self.history[-3:]  # Keep only recent
    
    def get_context(self):
        context = ""
        if self.summary:
            context += f"Previous conversation summary: {self.summary}\n\n"
        
        context += "Recent messages:\n"
        for msg in self.history:
            context += f"{msg['role']}: {msg['content']}\n"
        
        return context

# Example
bot = SummarizingChatbot()

for i in range(50):
    bot.add_message("user", f"Question about topic {i}")
    bot.add_message("assistant", f"Detailed answer about topic {i}")
    
    if bot.should_summarize():
        bot.summarize_history()
        print(f"Summarized! Summary tokens: {count_tokens(bot.summary)}")

```

**3. Semantic Search (Retrieve Relevant Context)**

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticChatbot:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.messages = []
        self.embeddings = []
    
    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
        
        # Create embedding
        embedding = self.encoder.encode(content)
        self.embeddings.append(embedding)
    
    def get_relevant_context(self, query, top_k=5):
        """Retrieve most relevant past messages."""
        # Encode query
        query_embedding = self.encoder.encode(query)
        
        # Calculate similarities
        similarities = [
            np.dot(query_embedding, emb) / 
            (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            for emb in self.embeddings
        ]
        
        # Get top k most relevant
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        relevant = [self.messages[i] for i in top_indices]
        
        return relevant
    
    def chat(self, user_query):
        # Get relevant history instead of all history
        relevant_context = self.get_relevant_context(user_query, top_k=3)
        
        # Build prompt
        context = "Relevant conversation history:\n"
        for msg in relevant_context:
            context += f"{msg['role']}: {msg['content']}\n"
        
        context += f"\nCurrent query: {user_query}"
        
        # Make API call with only relevant context
        return call_llm(context)

# Benefit: Constant context size regardless of conversation length!

```

### Cost Comparison: Context Strategies

```python
def compare_context_strategies():
    """Compare costs of different context management approaches."""
    
    # Simulation parameters
    num_exchanges = 50
    avg_message_tokens = 50
    
    # Strategy 1: Full history (no management)
    full_history_tokens = num_exchanges * 2 * avg_message_tokens  # 5,000 tokens
    full_cost = (full_history_tokens / 1000) * 0.03
    
    # Strategy 2: Sliding window (keep last 10 messages)
    window_tokens = 10 * 2 * avg_message_tokens  # 1,000 tokens
    window_cost = (window_tokens / 1000) * 0.03
    
    # Strategy 3: Summarization (200 token summary + 6 recent)
    summary_tokens = 200 + (6 * avg_message_tokens)  # 500 tokens
    summary_cost = (summary_tokens / 1000) * 0.03
    summary_cost += 0.01  # One-time cost to generate summary
    
    # Strategy 4: Semantic search (top 5 relevant)
    semantic_tokens = 5 * avg_message_tokens  # 250 tokens
    semantic_cost = (semantic_tokens / 1000) * 0.03
    
    print("Cost per request after 50 exchanges:")
    print(f"Full history: ${full_cost:.4f}")
    print(f"Sliding window: ${window_cost:.4f} ({(1-window_cost/full_cost)*100:.1f}% savings)")
    print(f"Summarization: ${summary_cost:.4f} ({(1-summary_cost/full_cost)*100:.1f}% savings)")
    print(f"Semantic search: ${semantic_cost:.4f} ({(1-semantic_cost/full_cost)*100:.1f}% savings)")
    
    # At scale (1 million requests)
    print(f"\nCost at 1 million requests:")
    print(f"Full history: ${full_cost * 1_000_000:,.0f}")
    print(f"Semantic search: ${semantic_cost * 1_000_000:,.0f}")
    print(f"Savings: ${(full_cost - semantic_cost) * 1_000_000:,.0f}")

compare_context_strategies()

# Output:
# Full history: $0.1500
# Sliding window: $0.0300 (80.0% savings)
# Summarization: $0.0250 (83.3% savings)
# Semantic search: $0.0075 (95.0% savings)
#
# Cost at 1 million requests:
# Full history: $150,000
# Semantic search: $7,500
# Savings: $142,500

```

---

## Part 3: GPU Hours and Utilization

### Understanding GPU Costs

**GPU Pricing (Example: AWS):**

```
NVIDIA A100 (40GB):
- On-demand: $4.10 per hour
- 1-year reserved: $2.40 per hour
- 3-year reserved: $1.60 per hour

NVIDIA V100 (16GB):
- On-demand: $3.06 per hour
- Reserved: ~$1.20-1.80 per hour

Cost breakdown:
Idle GPU: Still costs full price!
50% utilized: Still costs full price!
100% utilized: Same price, but 2× throughput

Lesson: Utilization matters!

```

### GPU Utilization Metrics

```python
import time
import torch

class GPUMonitor:
    def __init__(self):
        self.start_time = None
        self.active_time = 0
        self.idle_time = 0
    
    def start_task(self):
        if self.start_time is None:
            self.start_time = time.time()
        self.task_start = time.time()
    
    def end_task(self):
        task_duration = time.time() - self.task_start
        self.active_time += task_duration
    
    def wait(self, duration):
        time.sleep(duration)
        self.idle_time += duration
    
    def get_utilization(self):
        total_time = time.time() - self.start_time
        return (self.active_time / total_time) * 100 if total_time > 0 else 0
    
    def get_stats(self):
        total_time = time.time() - self.start_time
        return {
            "total_time": total_time,
            "active_time": self.active_time,
            "idle_time": self.idle_time,
            "utilization": self.get_utilization(),
            "wasted_cost": (self.idle_time / 3600) * 4.10  # A100 hourly rate
        }

# Example: Sequential processing
monitor = GPUMonitor()

for i in range(100):
    monitor.start_task()
    # Process single query (0.1 seconds)
    time.sleep(0.1)
    monitor.end_task()
    
    # Wait for next query (0.9 seconds)
    monitor.wait(0.9)

stats = monitor.get_stats()
print(f"GPU Utilization: {stats['utilization']:.1f}%")
print(f"Wasted cost: ${stats['wasted_cost']:.2f}")

# Output:
# GPU Utilization: 10.0%
# Wasted cost: $3.69
# (90% of GPU time wasted!)

```

### Batch Processing for GPU Efficiency

**Problem:** Processing queries one at a time wastes GPU capacity

```
Sequential processing:
Query 1: ████░░░░░░░░░░░░░░░░ (5% GPU used)
Query 2: ████░░░░░░░░░░░░░░░░ (5% GPU used)
Query 3: ████░░░░░░░░░░░░░░░░ (5% GPU used)
...

GPU utilization: 5%
Throughput: 10 queries/second
Cost per query: $0.041 (full GPU hour / 10 QPS)

```

**Solution:** Batch processing

```
Batch processing (32 queries):
Batch 1: ████████████████████ (90% GPU used)
Batch 2: ████████████████████ (90% GPU used)
Batch 3: ████████████████████ (90% GPU used)
...

GPU utilization: 90%
Throughput: 180 queries/second (18× faster!)
Cost per query: $0.0023 (full GPU hour / 180 QPS)

Savings: 95%

```

### Implementing Batching

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

class BatchedInference:
    def __init__(self, model_name, batch_size=32):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.batch_size = batch_size
        self.model.eval()
        
        # Move to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
    
    def generate_sequential(self, prompts):
        """Process prompts one at a time (inefficient)."""
        start = time.time()
        results = []
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=50)
            
            result = self.tokenizer.decode(outputs[0])
            results.append(result)
        
        duration = time.time() - start
        return results, duration
    
    def generate_batched(self, prompts):
        """Process prompts in batches (efficient)."""
        start = time.time()
        results = []
        
        # Process in batches
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i+self.batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate for entire batch at once
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=50)
            
            # Decode batch results
            batch_results = [
                self.tokenizer.decode(output) 
                for output in outputs
            ]
            results.extend(batch_results)
        
        duration = time.time() - start
        return results, duration

# Comparison
model = BatchedInference("gpt2", batch_size=32)

prompts = ["Tell me about AI"] * 100

# Sequential
seq_results, seq_time = model.generate_sequential(prompts)
print(f"Sequential: {seq_time:.2f}s ({len(prompts)/seq_time:.1f} QPS)")

# Batched
batch_results, batch_time = model.generate_batched(prompts)
print(f"Batched: {batch_time:.2f}s ({len(prompts)/batch_time:.1f} QPS)")

print(f"Speedup: {seq_time/batch_time:.1f}×")

# Output:
# Sequential: 50.0s (2.0 QPS)
# Batched: 5.0s (20.0 QPS)
# Speedup: 10.0×

```

### Optimal Batch Size Selection

```python
def find_optimal_batch_size(model, sample_prompts, gpu_memory_gb=16):
    """Find optimal batch size for GPU."""
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    results = []
    
    for batch_size in batch_sizes:
        try:
            # Test this batch size
            batch = sample_prompts[:batch_size]
            
            start = time.time()
            model.generate_batched(batch)
            duration = time.time() - start
            
            throughput = batch_size / duration
            
            results.append({
                "batch_size": batch_size,
                "duration": duration,
                "throughput": throughput,
                "efficiency": throughput / batch_size  # QPS per query in batch
            })
            
            print(f"Batch size {batch_size:3d}: {throughput:6.1f} QPS")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch size {batch_size:3d}: OOM (too large)")
                break
            raise
    
    # Find batch size with best throughput
    best = max(results, key=lambda x: x["throughput"])
    print(f"\nOptimal batch size: {best['batch_size']}")
    print(f"Throughput: {best['throughput']:.1f} QPS")
    
    return best["batch_size"]

# Example output:
# Batch size   1:   10.0 QPS
# Batch size   2:   18.0 QPS
# Batch size   4:   32.0 QPS
# Batch size   8:   55.0 QPS
# Batch size  16:   85.0 QPS
# Batch size  32:  120.0 QPS
# Batch size  64:  130.0 QPS  ← Diminishing returns
# Batch size 128: OOM (too large)
#
# Optimal batch size: 64

```

---

## Part 4: Batching Strategies and Tradeoffs

### Latency vs Throughput Tradeoff

**Without Batching:**

```
Request arrives → Process immediately → Return result

Latency per request: 100ms
Throughput: 10 QPS
User experience: Instant response ✓

```

**With Batching:**

```
Collect requests for 500ms → Process batch → Return results

Latency per request: 100ms (processing) + 500ms (waiting) = 600ms
Throughput: 64 QPS (if batch_size=32)
User experience: 500ms delay ✗

```

**The Tradeoff:**

```
                  Latency    Throughput    Cost/Query
No batching       Low (100ms)    Low (10 QPS)     High ($0.041)
Small batch (8)   Medium (200ms) Medium (40 QPS)  Medium ($0.010)
Large batch (64)  High (600ms)   High (100 QPS)   Low ($0.004)

Choose based on your requirements:
- Real-time chat: Low batching (or none)
- Bulk processing: Large batching
- API service: Medium batching with timeout

```

### Dynamic Batching

**Strategy:** Collect queries until batch is full OR timeout occurs

```python
import asyncio
from collections import deque
import time

class DynamicBatcher:
    def __init__(self, model, batch_size=32, max_wait_ms=100):
        self.model = model
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = deque()
        self.processing = False
    
    async def add_request(self, prompt):
        """Add request to batch queue."""
        future = asyncio.Future()
        self.queue.append((prompt, future))
        
        # Start processing if not already running
        if not self.processing:
            asyncio.create_task(self.process_batch())
        
        # Wait for result
        return await future
    
    async def process_batch(self):
        """Process accumulated batch."""
        self.processing = True
        start_time = time.time()
        
        while True:
            # Check if should process now
            batch_full = len(self.queue) >= self.batch_size
            timeout_reached = (time.time() - start_time) * 1000 >= self.max_wait_ms
            
            if batch_full or (timeout_reached and len(self.queue) > 0):
                # Extract batch
                batch_size = min(len(self.queue), self.batch_size)
                batch = [self.queue.popleft() for _ in range(batch_size)]
                
                # Process
                prompts = [item[0] for item in batch]
                results = self.model.generate_batched(prompts)
                
                # Return results to waiting futures
                for (prompt, future), result in zip(batch, results):
                    future.set_result(result)
                
                # Reset timer
                start_time = time.time()
            
            # Check if done
            if len(self.queue) == 0:
                self.processing = False
                break
            
            # Small sleep to accumulate more requests
            await asyncio.sleep(0.01)

# Usage
async def main():
    batcher = DynamicBatcher(model, batch_size=32, max_wait_ms=100)
    
    # Simulate requests arriving
    tasks = []
    for i in range(100):
        task = asyncio.create_task(batcher.add_request(f"Query {i}"))
        tasks.append(task)
        
        # Random arrival times
        await asyncio.sleep(random.uniform(0, 0.02))
    
    # Wait for all results
    results = await asyncio.gather(*tasks)
    print(f"Processed {len(results)} requests")

asyncio.run(main())

```

### Adaptive Batch Sizing

```python
class AdaptiveBatcher:
    def __init__(self, model):
        self.model = model
        self.batch_size = 8  # Start small
        self.min_batch = 4
        self.max_batch = 128
        
        # Performance tracking
        self.latencies = deque(maxlen=100)
        self.throughputs = deque(maxlen=100)
    
    def adjust_batch_size(self):
        """Adapt batch size based on performance."""
        if len(self.latencies) < 10:
            return  # Not enough data
        
        avg_latency = sum(self.latencies) / len(self.latencies)
        avg_throughput = sum(self.throughputs) / len(self.throughputs)
        
        # If latency too high, reduce batch size
        if avg_latency > 500:  # ms
            self.batch_size = max(self.min_batch, self.batch_size // 2)
            print(f"Reducing batch size to {self.batch_size} (latency: {avg_latency:.0f}ms)")
        
        # If latency acceptable and throughput can improve, increase
        elif avg_latency < 200 and self.batch_size < self.max_batch:
            self.batch_size = min(self.max_batch, self.batch_size * 2)
            print(f"Increasing batch size to {self.batch_size} (throughput: {avg_throughput:.0f} QPS)")
    
    def process_batch(self, prompts):
        """Process batch and track metrics."""
        start = time.time()
        
        results = self.model.generate_batched(prompts[:self.batch_size])
        
        duration = time.time() - start
        latency_ms = duration * 1000
        throughput = len(prompts) / duration
        
        # Track metrics
        self.latencies.append(latency_ms)
        self.throughputs.append(throughput)
        
        # Periodically adjust
        if len(self.latencies) % 10 == 0:
            self.adjust_batch_size()
        
        return results

```

---

## Part 5: Cost-Performance Tradeoffs

### Model Selection Strategies

**Cost vs Quality:**

```python
def route_to_model(query, complexity_threshold=0.7):
    """Route query to appropriate model based on complexity."""
    
    # Simple complexity heuristic
    complexity_score = calculate_complexity(query)
    
    if complexity_score < complexity_threshold:
        # Use cheaper model
        model = "gpt-3.5-turbo"
        cost_multiplier = 1.0
    else:
        # Use more capable model
        model = "gpt-4"
        cost_multiplier = 30.0
    
    return model, cost_multiplier

def calculate_complexity(query):
    """Estimate query complexity."""
    # Factors:
    # - Length
    # - Technical terms
    # - Multiple questions
    # - Ambiguity
    
    score = 0.0
    
    # Length factor
    words = len(query.split())
    if words > 50:
        score += 0.3
    elif words > 20:
        score += 0.1
    
    # Technical terms
    technical_terms = ["algorithm", "quantum", "derivative", "synthesis"]
    if any(term in query.lower() for term in technical_terms):
        score += 0.4
    
    # Multiple questions
    if query.count("?") > 1:
        score += 0.2
    
    # Requires reasoning
    reasoning_words = ["why", "how", "explain", "compare"]
    if any(word in query.lower() for word in reasoning_words):
        score += 0.3
    
    return min(score, 1.0)

# Example usage
queries = [
    "What is 2+2?",  # Simple
    "Explain the quantum entanglement paradox",  # Complex
    "How does gradient descent work in neural networks?",  # Complex
    "What time is it?"  # Simple
]

for query in queries:
    model, cost_mult = route_to_model(query)
    print(f"Query: {query[:50]}...")
    print(f"  → Route to {model} (cost: {cost_mult}×)")
    print()

# Output:
# Query: What is 2+2?
#   → Route to gpt-3.5-turbo (cost: 1.0×)
#
# Query: Explain the quantum entanglement paradox
#   → Route to gpt-4 (cost: 30.0×)

```

### Caching Strategies

```python
import hashlib
from redis import Redis
import json

class ResponseCache:
    def __init__(self, ttl=3600):
        self.cache = Redis()
        self.ttl = ttl  # Time to live in seconds
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.cost_saved = 0.0
    
    def get_cache_key(self, prompt):
        """Generate cache key from prompt."""
        # Normalize prompt
        normalized = prompt.lower().strip()
        
        # Create hash
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get(self, prompt):
        """Get cached response if exists."""
        key = self.get_cache_key(prompt)
        cached = self.cache.get(key)
        
        if cached:
            self.hits += 1
            return json.loads(cached)
        
        self.misses += 1
        return None
    
    def set(self, prompt, response, cost):
        """Cache response."""
        key = self.get_cache_key(prompt)
        value = json.dumps({
            "response": response,
            "cost": cost,
            "timestamp": time.time()
        })
        
        self.cache.setex(key, self.ttl, value)
    
    def get_with_generation(self, prompt, generate_fn):
        """Get from cache or generate and cache."""
        # Try cache first
        cached = self.get(prompt)
        if cached:
            self.cost_saved += cached["cost"]
            return cached["response"], 0.0  # No cost for cache hit
        
        # Generate
        response = generate_fn(prompt)
        cost = calculate_cost(prompt, response)
        
        # Cache for future
        self.set(prompt, response, cost)
        
        return response, cost
    
    def get_stats(self):
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "cost_saved": self.cost_saved
        }

# Usage example
cache = ResponseCache(ttl=3600)

common_queries = [
    "What is Python?",
    "How do I install NumPy?",
    "What is machine learning?"
]

# Simulate traffic with repeated queries
for _ in range(1000):
    query = random.choice(common_queries)
    response, cost = cache.get_with_generation(query, call_llm)
    
    # 70% of queries are from top 3 common queries
    # These will be cache hits after first time

stats = cache.get_stats()
print(f"Cache hit rate: {stats['hit_rate']:.1f}%")
print(f"Cost saved: ${stats['cost_saved']:.2f}")

# Output:
# Cache hit rate: 69.7%
# Cost saved: $27.88
# (Only 30.3% of queries hit the expensive LLM!)

```

### Quality vs Cost Monitoring

```python
class QualityMonitor:
    def __init__(self):
        self.responses = []
    
    def log_response(self, query, response, model, cost, quality_score):
        """Log response with quality metrics."""
        self.responses.append({
            "query": query,
            "response": response,
            "model": model,
            "cost": cost,
            "quality": quality_score
        })
    
    def analyze_tradeoffs(self):
        """Analyze cost-quality tradeoffs."""
        # Group by model
        by_model = {}
        for resp in self.responses:
            model = resp["model"]
            if model not in by_model:
                by_model[model] = {"costs": [], "qualities": []}
            
            by_model[model]["costs"].append(resp["cost"])
            by_model[model]["qualities"].append(resp["quality"])
        
        # Calculate averages
        print("Cost-Quality Analysis:")
        for model, data in by_model.items():
            avg_cost = sum(data["costs"]) / len(data["costs"])
            avg_quality = sum(data["qualities"]) / len(data["qualities"])
            
            print(f"\n{model}:")
            print(f"  Average cost: ${avg_cost:.4f}")
            print(f"  Average quality: {avg_quality:.2f}/5.0")
            print(f"  Cost per quality point: ${avg_cost/avg_quality:.4f}")
        
        # Find optimal model
        best_value = None
        best_ratio = float('inf')
        
        for model, data in by_model.items():
            avg_cost = sum(data["costs"]) / len(data["costs"])
            avg_quality = sum(data["qualities"]) / len(data["qualities"])
            ratio = avg_cost / avg_quality
            
            if ratio < best_ratio:
                best_ratio = ratio
                best_value = model
        
        print(f"\nBest value model: {best_value}")
        print(f"Cost per quality point: ${best_ratio:.4f}")

# Example output:
# Cost-Quality Analysis:
#
# gpt-3.5-turbo:
#   Average cost: $0.0015
#   Average quality: 3.80/5.0
#   Cost per quality point: $0.0004
#
# gpt-4:
#   Average cost: $0.0450
#   Average quality: 4.60/5.0
#   Cost per quality point: $0.0098
#
# Best value model: gpt-3.5-turbo
# Cost per quality point: $0.0004

```

---

## Part 6: Best Practices and Monitoring

### Comprehensive Cost Tracking

```python
class LLMCostTracker:
    def __init__(self):
        self.requests = []
        self.daily_totals = {}
    
    def log_request(self, model, input_tokens, output_tokens, latency_ms):
        """Log a request with all relevant metrics."""
        # Calculate cost
        pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002}
        }
        
        input_cost = (input_tokens / 1000) * pricing[model]["input"]
        output_cost = (output_tokens / 1000) * pricing[model]["output"]
        total_cost = input_cost + output_cost
        
        # Log
        timestamp = time.time()
        date = time.strftime("%Y-%m-%d", time.localtime(timestamp))
        
        request = {
            "timestamp": timestamp,
            "date": date,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "cost": total_cost,
            "latency_ms": latency_ms
        }
        
        self.requests.append(request)
        
        # Update daily total
        if date not in self.daily_totals:
            self.daily_totals[date] = 0.0
        self.daily_totals[date] += total_cost
    
    def get_summary(self, last_n_days=7):
        """Get cost summary."""
        cutoff = time.time() - (last_n_days * 86400)
        recent = [r for r in self.requests if r["timestamp"] >= cutoff]
        
        if not recent:
            return "No data"
        
        total_cost = sum(r["cost"] for r in recent)
        total_requests = len(recent)
        total_tokens = sum(r["total_tokens"] for r in recent)
        avg_latency = sum(r["latency_ms"] for r in recent) / total_requests
        
        # By model
        by_model = {}
        for r in recent:
            model = r["model"]
            if model not in by_model:
                by_model[model] = {"count": 0, "cost": 0.0}
            by_model[model]["count"] += 1
            by_model[model]["cost"] += r["cost"]
        
        print(f"Summary (Last {last_n_days} days):")
        print(f"  Total requests: {total_requests:,}")
        print(f"  Total cost: ${total_cost:.2f}")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Avg latency: {avg_latency:.0f}ms")
        print(f"  Cost per request: ${total_cost/total_requests:.4f}")
        print(f"\nBy model:")
        for model, stats in by_model.items():
            print(f"  {model}:")
            print(f"    Requests: {stats['count']:,} ({stats['count']/total_requests*100:.1f}%)")
            print(f"    Cost: ${stats['cost']:.2f} ({stats['cost']/total_cost*100:.1f}%)")
        
        return {
            "total_cost": total_cost,
            "total_requests": total_requests,
            "by_model": by_model
        }

# Usage
tracker = LLMCostTracker()

# Log some requests
tracker.log_request("gpt-4", 1000, 500, 1200)
tracker.log_request("gpt-3.5-turbo", 500, 200, 300)
# ... log thousands more ...

# Get summary
tracker.get_summary(last_n_days=7)

```

### Alerts and Budget Management

```python
class BudgetManager:
    def __init__(self, daily_budget=100, monthly_budget=2000):
        self.daily_budget = daily_budget
        self.monthly_budget = monthly_budget
        self.tracker = LLMCostTracker()
    
    def check_budget(self):
        """Check if budget exceeded."""
        today = time.strftime("%Y-%m-%d")
        month = time.strftime("%Y-%m")
        
        # Daily check
        daily_spend = self.tracker.daily_totals.get(today, 0.0)
        if daily_spend > self.daily_budget:
            self.alert(f"Daily budget exceeded: ${daily_spend:.2f} > ${self.daily_budget}")
        elif daily_spend > self.daily_budget * 0.8:
            self.alert(f"Warning: 80% of daily budget used (${daily_spend:.2f})")
        
        # Monthly check
        monthly_spend = sum(
            cost for date, cost in self.tracker.daily_totals.items()
            if date.startswith(month)
        )
        
        if monthly_spend > self.monthly_budget:
            self.alert(f"Monthly budget exceeded: ${monthly_spend:.2f} > ${self.monthly_budget}")
        elif monthly_spend > self.monthly_budget * 0.8:
            self.alert(f"Warning: 80% of monthly budget used (${monthly_spend:.2f})")
    
    def alert(self, message):
        """Send alert (implement your notification system)."""
        print(f"ALERT: {message}")
        # Send email, Slack message, etc.
    
    def project_monthly_cost(self):
        """Project end-of-month cost based on current trends."""
        month = time.strftime("%Y-%m")
        current_day = int(time.strftime("%d"))
        
        monthly_spend = sum(
            cost for date, cost in self.tracker.daily_totals.items()
            if date.startswith(month)
        )
        
        avg_daily = monthly_spend / current_day
        days_in_month = 30  # Approximate
        projected = avg_daily * days_in_month
        
        print(f"Monthly projection:")
        print(f"  Current spend (day {current_day}): ${monthly_spend:.2f}")
        print(f"  Projected end-of-month: ${projected:.2f}")
        print(f"  Budget: ${self.monthly_budget}")
        
        if projected > self.monthly_budget:
            overage = projected - self.monthly_budget
            print(f"  WARNING: Projected overage: ${overage:.2f}")
        
        return projected

# Usage
budget_mgr = BudgetManager(daily_budget=100, monthly_budget=2000)

# Check periodically
budget_mgr.check_budget()
budget_mgr.project_monthly_cost()

```

---

## Complete Example: Optimized LLM Application

```python
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer
import tiktoken
from redis import Redis
import time

class OptimizedLLMApp:
    def __init__(self):
        # Components
        self.cache = ResponseCache(ttl=3600)
        self.batcher = DynamicBatcher(batch_size=32, max_wait_ms=100)
        self.tracker = LLMCostTracker()
        self.budget_mgr = BudgetManager(daily_budget=50, monthly_budget=1000)
        
        # Models
        self.cheap_model = "gpt-3.5-turbo"
        self.expensive_model = "gpt-4"
    
    async def process_query(self, query):
        """Process query with all optimizations."""
        start_time = time.time()
        
        # 1. Check cache first
        cached = self.cache.get(query)
        if cached:
            latency = (time.time() - start_time) * 1000
            # Log cache hit (zero cost)
            self.tracker.log_request("cache", 0, 0, latency)
            return cached["response"]
        
        # 2. Route to appropriate model
        model = self.route_query(query)
        
        # 3. Optimize context
        optimized_query = self.optimize_context(query)
        
        # 4. Count tokens
        input_tokens = count_tokens(optimized_query, model)
        
        # 5. Generate response (batched)
        response = await self.batcher.add_request(optimized_query)
        output_tokens = count_tokens(response, model)
        
        # 6. Cache response
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        self.cache.set(query, response, cost)
        
        # 7. Log metrics
        latency = (time.time() - start_time) * 1000
        self.tracker.log_request(model, input_tokens, output_tokens, latency)
        
        # 8. Check budget
        self.budget_mgr.check_budget()
        
        return response
    
    def route_query(self, query):
        """Route to cheap or expensive model based on complexity."""
        complexity = calculate_complexity(query)
        return self.expensive_model if complexity > 0.7 else self.cheap_model
    
    def optimize_context(self, query):
        """Optimize query to reduce tokens."""
        # Truncate if too long
        max_tokens = 2000
        if count_tokens(query) > max_tokens:
            query = truncate_to_token_limit(query, max_tokens)
        
        # Remove redundant whitespace
        query = " ".join(query.split())
        
        return query
    
    def calculate_cost(self, model, input_tokens, output_tokens):
        """Calculate cost for request."""
        pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002}
        }
        
        input_cost = (input_tokens / 1000) * pricing[model]["input"]
        output_cost = (output_tokens / 1000) * pricing[model]["output"]
        
        return input_cost + output_cost
    
    async def process_batch(self, queries):
        """Process multiple queries efficiently."""
        tasks = [self.process_query(q) for q in queries]
        return await asyncio.gather(*tasks)
    
    def get_stats(self):
        """Get comprehensive statistics."""
        print("="*60)
        print("PERFORMANCE METRICS")
        print("="*60)
        
        # Cost summary
        self.tracker.get_summary(last_n_days=7)
        
        # Cache stats
        cache_stats = self.cache.get_stats()
        print(f"\nCache performance:")
        print(f"  Hit rate: {cache_stats['hit_rate']:.1f}%")
        print(f"  Cost saved: ${cache_stats['cost_saved']:.2f}")
        
        # Budget status
        self.budget_mgr.project_monthly_cost()

# Usage
app = OptimizedLLMApp()

# Process queries
queries = ["What is Python?"] * 100  # Many duplicate queries
results = asyncio.run(app.process_batch(queries))

# Get statistics
app.get_stats()

# Expected output:
# - First query: Expensive (cache miss)
# - Remaining 99: Free (cache hits)
# - High cache hit rate (99%)
# - Low total cost
# - Good throughput due to batching

```

---

## Key Takeaways

**Token Economics:**

- Count every token - they all cost money
- 1 word ≈ 1.3 tokens average
- Input and output tokens priced separately
- Small optimizations × millions of requests = huge savings

**Context Management:**

- Sliding window: Keep recent history only
- Summarization: Compress old history
- Semantic search: Retrieve relevant context
- Can save 90%+ on context costs

**GPU Utilization:**

- Batch processing increases utilization
- Higher utilization = lower cost per query
- Can achieve 10-30× speedup with batching
- Monitor and optimize continuously

**Cost-Performance Tradeoffs:**

- Model routing: Use cheap models when possible
- Caching: Avoid redundant computation
- Quality vs cost: Monitor both
- Latency vs throughput: Balance based on use case

**Monitoring:**

- Track costs in real-time
- Set budgets and alerts
- Project monthly spending
- Optimize based on data

---

**End of Lecture Notes**

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