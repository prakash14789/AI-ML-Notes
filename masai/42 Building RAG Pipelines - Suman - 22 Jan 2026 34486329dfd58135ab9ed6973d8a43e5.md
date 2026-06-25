# 42. Building RAG Pipelines - Suman - 22 Jan 2026

# Building RAG Pipelines: Embeddings, Hybrid Search & Evaluation

## [In-Class Resources](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/df6e5a6f-aac8-4e7e-9e94-2fcc712b7336/yhoGZwkjvwFwoBGK.pptx)

**Prerequisites:** Understanding of vector databases and basic RAG concepts, familiarity with embeddings and similarity search, basic Python programming skills.

**Time to complete:** 35-45 minutes

**What you'll be able to do:**

- Implement efficient batch embedding generation for large document collections
- Build hybrid search systems combining semantic and keyword approaches
- Design and execute evaluation harnesses for RAG pipeline quality
- Optimize pipeline components for production deployment

---

## 1. Introduction: What is a RAG Pipeline and Why Should You Care?

### Core Definition

A **RAG pipeline** is the complete system that takes user queries, retrieves relevant documents, and generates grounded responses. Beyond the basic components (chunker, embedder, vector store, LLM), production pipelines include batch processing, hybrid retrieval, re-ranking, caching, and continuous evaluation.

**Batch embedding generation** is the process of converting large document collections into vectors efficiently, using batching, parallelization, and GPU acceleration to handle millions of documents in reasonable time.

**Hybrid search** combines dense retrieval (embeddings) with sparse retrieval (BM25/TF-IDF) to capture both semantic meaning and exact keyword matches.

### A Simple Analogy

Think of a RAG pipeline like a research assistant. Basic RAG is like asking them to find relevant books. A production pipeline adds: batch processing (they organize the library once, efficiently), hybrid search (they check both topic relevance AND specific keywords you mentioned), and evaluation (they track how often their findings actually help you).

**Limitation:** Unlike a human assistant who learns your preferences over time, RAG pipelines need explicit configuration changes to improve—they don't self-optimize without evaluation feedback.

### Why This Matters to You

**Problem it solves:** Basic RAG demos work on small datasets but break at scale. Production systems need to handle millions of documents, balance precision vs recall, and continuously measure quality.

**What you'll gain:**

- Build RAG systems that scale to enterprise document collections
- Retrieve more relevant results by combining multiple search strategies
- Measure and improve pipeline quality systematically

**Real-world context:** Notion AI, Confluence search, and enterprise knowledge bases all use production RAG pipelines with these exact components.

---

## 2. The Foundation: Core Concepts Explained

### Concept A: Batch Embedding Generation

**Definition:** Batch embedding generation processes documents in groups rather than one-at-a-time, dramatically reducing overhead from model loading, API calls, and memory management. For large collections, batching can reduce embedding time by 10-50x.

**Key characteristics:**

- **Optimal batch size:** Typically 32-128 for local models, limited by GPU memory; API providers often have their own limits (e.g., 2048 texts for OpenAI)
- **Parallelization:** Multiple batches can process simultaneously across CPU cores or GPU streams
- **Checkpointing:** Save progress periodically so you don't lose work if the process fails midway

**A concrete example:**

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def batch_embed(texts, batch_size=64):
    """Embed texts in batches for efficiency."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

# Process 10,000 documents efficiently
all_embeddings = batch_embed(documents, batch_size=64)

```

**Common confusion:** Beginners often embed one document at a time in a loop. This wastes time on repeated model overhead and misses GPU parallelization opportunities.

---

### Concept B: Hybrid Search

**Definition:** Hybrid search combines dense retrieval (vector similarity) with sparse retrieval (keyword matching like BM25) to leverage the strengths of both: semantic understanding from embeddings and exact term matching from traditional search.

**How it relates to Batch Embedding:** Your embeddings power the dense retrieval component, while the original text powers the sparse component. Both operate on the same chunked documents.

**Key characteristics:**

- **Complementary strengths:** Dense catches "password reset" when user asks about "forgot login"; sparse catches exact product codes or technical terms that embeddings might miss
- **Score fusion:** Combine rankings using weighted sums, reciprocal rank fusion (RRF), or learned re-rankers
- **Query-dependent weighting:** Some queries benefit more from semantic (conceptual questions) vs sparse (specific terms)

**A concrete example:**

```python
from rank_bm25 import BM25Okapi
import numpy as np

# Sparse retrieval component
tokenized_docs = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

def hybrid_search(query, documents, embeddings, index, alpha=0.5, k=10):
    """Combine dense and sparse retrieval."""
    # Dense scores (cosine similarity)
    query_embedding = model.encode([query])
    distances, dense_indices = index.search(query_embedding, k * 2)
    dense_scores = 1 / (1 + distances[0])  # Convert distance to score

    # Sparse scores (BM25)
    sparse_scores = bm25.get_scores(query.lower().split())

    # Normalize and combine
    dense_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-6)
    sparse_norm = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min() + 1e-6)

    combined = alpha * dense_norm + (1 - alpha) * sparse_norm[dense_indices[0]]

    # Return top-k by combined score
    top_k = np.argsort(combined)[::-1][:k]
    return [(dense_indices[0][i], combined[i]) for i in top_k]

```

**Remember:** The optimal alpha (weighting) depends on your use case. Technical documentation with specific terms often benefits from higher sparse weight; conversational FAQs benefit from higher dense weight.

---

### Concept C: Evaluation Harness

**Definition:** An evaluation harness is a systematic framework for measuring RAG pipeline quality. It includes test datasets, metrics computation, result logging, and comparison across pipeline configurations.

**Key characteristics:**

- **Test dataset:** Question-answer pairs with known relevant documents
- **Retrieval metrics:** Recall@K, MRR (Mean Reciprocal Rank), NDCG (Normalized Discounted Cumulative Gain)
- **End-to-end metrics:** Answer correctness, grounding score, latency
- **A/B comparison:** Compare configurations side-by-side

**A concrete example:**

```python
def evaluate_retrieval(test_set, retrieval_fn, k=5):
    """Evaluate retrieval quality on a test set."""
    metrics = {'recall': [], 'mrr': []}

    for item in test_set:
        query = item['question']
        relevant_doc_ids = item['relevant_docs']

        # Get retrieval results
        results = retrieval_fn(query, k=k)
        retrieved_ids = [r['id'] for r in results]

        # Recall@K: Did we find any relevant doc?
        hits = len(set(retrieved_ids) & set(relevant_doc_ids))
        metrics['recall'].append(hits / len(relevant_doc_ids))

        # MRR: Rank of first relevant doc
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_doc_ids:
                metrics['mrr'].append(1 / rank)
                break
        else:
            metrics['mrr'].append(0)

    return {k: sum(v) / len(v) for k, v in metrics.items()}

```

---

### How These Concepts Work Together

The production pipeline flow: Documents → **Batch Embedding** (efficient vectorization) → Index → **Hybrid Search** (semantic + keyword) → Re-rank → LLM → Response. The **Evaluation Harness** measures quality at each stage, enabling data-driven optimization.

---

## 3. Seeing It in Action: Worked Examples

### Example 1: Efficient Batch Embedding with Checkpointing

**Scenario:** You have 500,000 documents to embed for a legal research platform. Processing one-at-a-time would take days; you need efficient batching with failure recovery.

**Our approach:** Batch process with checkpointing every 10,000 documents, using GPU acceleration and progress tracking.

**Step-by-step solution:**

```python
# Step 1: Setup with GPU if available
import torch
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Step 2: Checkpointed batch processing
def embed_with_checkpoints(documents, batch_size=128, checkpoint_every=10000,
                           checkpoint_dir='./checkpoints'):
    Path(checkpoint_dir).mkdir(exist_ok=True)
    all_embeddings = []

    # Check for existing checkpoint
    checkpoint_files = sorted(Path(checkpoint_dir).glob('checkpoint_*.pkl'))
    start_idx = 0
    if checkpoint_files:
        latest = checkpoint_files[-1]
        with open(latest, 'rb') as f:
            all_embeddings = pickle.load(f)
        start_idx = len(all_embeddings)
        print(f"Resuming from checkpoint at index {start_idx}")

    # Step 3: Process in batches
    for i in range(start_idx, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        embeddings = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        all_embeddings.extend(embeddings)

        # Save checkpoint
        if len(all_embeddings) % checkpoint_every == 0:
            with open(f'{checkpoint_dir}/checkpoint_{len(all_embeddings)}.pkl', 'wb') as f:
                pickle.dump(all_embeddings, f)
            print(f"Checkpoint saved at {len(all_embeddings)} documents")

    return np.array(all_embeddings)

```

**Output:**

```
Resuming from checkpoint at index 100000
Checkpoint saved at 110000 documents
Checkpoint saved at 120000 documents
...

```

**What just happened:** We processed 500K documents with automatic resume capability. If the process crashes at document 350,000, we only lose the last partial checkpoint, not all previous work.

**Check your understanding:** Why do we save checkpoints rather than processing everything in memory first?

---

### Example 2: Building a Hybrid Search System

**Scenario:** A technical documentation site notices that semantic search misses exact function names and error codes. Users searching for "RuntimeError: CUDA out of memory" get general GPU articles instead of the specific error page.

**What's different:** Pure semantic search understands concepts but can miss exact technical terms. We'll add BM25 to catch these.

**Solution:**

```python
# Step 1: Initialize both retrieval systems
from rank_bm25 import BM25Okapi
import faiss

# Dense index (from batch embeddings)
dimension = 384
index = faiss.IndexFlatIP(dimension)  # Inner product for cosine sim
faiss.normalize_L2(embeddings)  # Normalize for cosine
index.add(embeddings)

# Sparse index (BM25)
tokenized_corpus = [doc.lower().split() for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

# Step 2: Reciprocal Rank Fusion
def reciprocal_rank_fusion(rankings, k=60):
    """Combine multiple rankings using RRF."""
    fused_scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            if doc_id not in fused_scores:
                fused_scores[doc_id] = 0
            fused_scores[doc_id] += 1 / (k + rank + 1)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

# Step 3: Hybrid retrieval
def hybrid_retrieve(query, top_k=10):
    # Dense retrieval
    q_emb = model.encode([query])
    faiss.normalize_L2(q_emb)
    _, dense_ids = index.search(q_emb, top_k * 2)

    # Sparse retrieval
    sparse_scores = bm25.get_scores(query.lower().split())
    sparse_ids = np.argsort(sparse_scores)[::-1][:top_k * 2]

    # Fuse rankings
    fused = reciprocal_rank_fusion([dense_ids[0].tolist(), sparse_ids.tolist()])
    return [doc_id for doc_id, score in fused[:top_k]]

```

**Key lesson:** RRF is robust because it only uses rank positions, not raw scores that need calibration. A document ranked #1 by both systems gets higher combined score than one ranked #1 by only one system.

---

### Example 3: Production Evaluation Harness

**Background:** A customer support team deployed RAG but has no way to measure if it's working. They need systematic evaluation to catch regressions and compare improvements.

**The challenge:** Create an evaluation pipeline that measures retrieval quality, answer accuracy, and latency—and runs automatically on every pipeline change.

**The approach:** Build a test harness with golden test set, multiple metrics, and comparison reporting.

**Why this approach:** Without systematic evaluation, you can't tell if changes help or hurt. A/B comparisons with confidence intervals separate real improvements from noise.

**The outcome:**

```python
class RAGEvaluator:
    def __init__(self, test_set, retrieval_fn, generation_fn):
        self.test_set = test_set
        self.retrieve = retrieval_fn
        self.generate = generation_fn

    def run_evaluation(self, config_name="default"):
        results = []
        for item in self.test_set:
            start = time.time()

            # Retrieval phase
            retrieved = self.retrieve(item['question'], k=5)
            retrieval_time = time.time() - start

            # Generation phase
            response = self.generate(item['question'], retrieved)
            total_time = time.time() - start

            # Compute metrics
            results.append({
                'question': item['question'],
                'recall@5': self._compute_recall(retrieved, item['relevant_docs']),
                'mrr': self._compute_mrr(retrieved, item['relevant_docs']),
                'answer_correct': self._check_answer(response, item['expected_answer']),
                'latency_ms': total_time * 1000
            })

        return self._aggregate_results(results, config_name)

```

**Caution:** Golden test sets can become stale as your document collection changes. Schedule periodic reviews to ensure test questions still have valid relevant documents.

---

## 4. Common Pitfalls: What Can Go Wrong and How to Avoid It

- **The Mistake:** Embedding documents one-at-a-time in a loop

**Why It's a Problem:** Each call has overhead for model setup, memory allocation, and data transfer. For 100K documents, you might spend 10x more time on overhead than actual computation.
**The Right Approach:** Batch encode with sizes of 32-128, matching your GPU memory. Use `model.encode(batch_list)` instead of looping.
**Why This Works:** Batching amortizes fixed costs and enables parallel computation on GPU tensor cores.

---

- **The Mistake:** Using only semantic search for technical documentation

**Why It's a Problem:** Embeddings may not capture exact error codes, function names, or version numbers. "numpy 1.24.0 compatibility" might retrieve general numpy docs instead of version-specific pages.
**The Right Approach:** Implement hybrid search with BM25 for exact matching plus embeddings for semantic understanding.
**Why This Works:** BM25 excels at exact term matching; embeddings excel at conceptual similarity. Together they cover both needs.

---

- **The Mistake:** No evaluation framework—just "it seems to work"

**Why It's a Problem:** You can't measure improvement, catch regressions, or make data-driven decisions about pipeline changes.
**The Right Approach:** Build a golden test set with at least 50-100 examples. Run evaluations on every significant change.
**Why This Works:** Systematic measurement turns gut feelings into evidence. You can prove that change X improved recall by 12%.

**If you're stuck:** Start with a small test set (20 questions) and simple metrics (Recall@5). Expand as you learn what matters for your use case.

---

## 5. Your Turn: Practice & Self-Assessment

### Practice Task (Estimated: 15-20 minutes)

**The Challenge:** Build a complete evaluation harness for a RAG pipeline on documentation of your choice.

**Specifications:**

- Create a test set with 15 questions and their known relevant document IDs
- Implement Recall@5 and MRR metrics
- Compare performance between dense-only and hybrid search
- Generate a comparison report with confidence intervals

**Hint:** Start by manually finding 15 question-answer pairs in your docs. For each, note which chunk(s) contain the answer—those are your relevant document IDs.

**Extension (optional):** Add latency tracking and plot the precision-latency tradeoff across different K values.

---

### Check Your Understanding

1. **Explanation question:** Why does batch size matter for embedding generation, and what factors determine the optimal batch size?
2. **Application question:** Your hybrid search has alpha=0.5 but users searching for error codes get poor results. How would you adjust?
3. **Error analysis:** A pipeline shows 95% Recall@10 but users still complain about irrelevant results. What might explain this gap?
4. **Transfer question:** How would you adapt an evaluation harness designed for English docs to handle a multilingual document collection?

**Answers & Explanations:**

1. 
Batch size matters because it determines how many documents share the overhead of model invocation and GPU data transfer. Optimal size balances: GPU memory (larger = better until OOM), model architecture (transformers often have sweet spots around 32-64), and API limits (OpenAI caps at 2048). Experiment to find your specific optimum.

2. 
Increase sparse weight (lower alpha, e.g., 0.3 for dense, 0.7 for sparse) for queries with exact technical terms. Better yet, implement query classification to dynamically adjust alpha based on whether the query contains technical codes vs natural language.

3. 
Possible causes: (a) Test set doesn't represent real user queries, (b) Top-10 includes the relevant doc but at position 8-10 where users don't look, (c) Relevant doc is retrieved but doesn't contain a complete answer, (d) Multiple relevant docs are needed but metric only counts presence. Analyze failure cases qualitatively.

4. 
Key adaptations: (a) Use multilingual embedding model for dense retrieval, (b) Use language-specific tokenizers for BM25, (c) Stratify test set by language to catch per-language quality issues, (d) Consider whether cross-lingual retrieval is needed (query in one language, docs in another).

---

### Self-Assessment Checklist

You've mastered this topic if you can:

- Implement batch embedding with checkpointing for large document sets
- Explain when and why hybrid search outperforms dense-only retrieval
- Build an evaluation harness with multiple metrics
- Interpret evaluation results and identify where pipelines fail
- Make data-driven decisions about pipeline configurations
- Adapt these patterns to different document types and scales

**If you checked fewer than 5 boxes:** Focus on the worked examples—implement them yourself before moving to more complex scenarios.

---

## 6. Consolidation: Key Takeaways & Next Steps

### The Essential Ideas

- **Batch processing enables scale:** Efficient embedding generation is the difference between processing 10K docs in minutes vs. days.
- **Hybrid search catches what semantics miss:** Combine BM25 and embeddings to handle both conceptual queries and exact technical terms.
- **Evaluation makes improvement measurable:** Without systematic metrics, you're guessing whether changes help.

### Mental Model Check

By now, you should think of RAG pipelines as: a data processing system with multiple stages, each measurable and optimizable, where the right combination of dense and sparse retrieval—guided by evaluation data—delivers the best results.

### What You Can Now Do

You can build production-grade RAG pipelines that handle large document collections efficiently, retrieve effectively for diverse query types, and measure quality systematically.

### Next Steps

**To deepen this knowledge:** Implement these patterns on a dataset 10x larger than your practice set. Real scaling issues often emerge only at scale.

**To build on this:** Learn about re-ranking with cross-encoders and query expansion for even better retrieval.

**Additional resources:** Sentence-Transformers documentation for embedding optimization, BEIR benchmark for evaluation methodology.

---

## Quick Reference Card

Component | Purpose | Key Configuration
Batch Embedder | Efficient vectorization | batch_size=64-128, checkpoints
BM25 Index | Sparse retrieval | Tokenizer matters for domain
Dense Index | Semantic retrieval | Normalize for cosine similarity
Hybrid Fusion | Combine rankings | RRF with k=60, or weighted sum
Eval Harness | Measure quality | Recall@K, MRR, latency

---

**Questions or stuck?** The sentence-transformers library has excellent examples for batch processing and evaluation patterns.

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