# 43. RAG Evaluation & Tuning - Suman - 24 Jan 2026

# RAG Evaluation & Tuning: Retrieval Scores, Ground Truth, and Hallucination Mitigation

**Prerequisites:** Understanding of RAG pipeline components, familiarity with retrieval metrics (Recall@K, MRR), basic understanding of LLM generation.

**What you'll be able to do:**

- Implement comprehensive retrieval scoring and evaluation
- Compare generated answers against ground truth systematically
- Identify and fix hallucination patterns in RAG systems
- Design evaluation pipelines that catch quality regressions

---

## 1. Introduction: What is RAG Evaluation and Why Should You Care?

### Core Definition

**RAG evaluation** is the systematic measurement of how well your retrieval-augmented generation system performs across two critical dimensions: (1) retrieval quality—are we finding the right documents? and (2) generation quality—are we producing accurate, grounded answers?

**Ground truth comparison** involves measuring how close generated answers are to known correct answers, using metrics like exact match, semantic similarity, and factual accuracy scoring.

**Hallucination detection** identifies when the LLM generates information not supported by retrieved context, a critical failure mode in production RAG systems.

### A Simple Analogy

Think of RAG evaluation like grading a student's research paper. Retrieval evaluation checks if they found the right sources. Ground truth comparison checks if their conclusions match established facts. Hallucination detection checks if they made up citations or invented "facts" not in their sources.

**Limitation:** Unlike human graders who understand nuance, automated evaluation can miss subtle correctness issues or incorrectly flag valid paraphrasing as errors.

### Why This Matters to You

**Problem it solves:** Without evaluation, you can't tell if your RAG system is improving or degrading. A pipeline change that seems good might actually introduce subtle errors that hurt user experience.

**What you'll gain:**

- Confidence that your system works correctly before deployment
- Ability to catch and fix hallucinations before users see them
- Data-driven approach to tuning pipeline parameters

**Real-world context:** Enterprise AI deployments at legal, medical, and financial firms require rigorous evaluation because errors have real consequences. Microsoft, Google, and OpenAI all invest heavily in evaluation frameworks.

---

## 2. The Foundation: Core Concepts Explained

### Concept A: Retrieval Scoring

**Definition:** Retrieval scoring measures how effectively your system finds relevant documents. Key metrics include precision (what fraction of retrieved docs are relevant), recall (what fraction of relevant docs are retrieved), and ranking quality (are relevant docs ranked high?).

**Key characteristics:**

- **Recall@K:** Fraction of relevant documents in top-K results. High recall means you're not missing relevant content.
- **Precision@K:** Fraction of top-K results that are relevant. High precision means you're not flooding the LLM with noise.
- **MRR (Mean Reciprocal Rank):** Average of 1/rank for first relevant doc. High MRR means relevant content appears early.
- **NDCG (Normalized Discounted Cumulative Gain):** Considers graded relevance and position. Ideal when some docs are "more relevant" than others.

**A concrete example:**

```python
def compute_retrieval_metrics(retrieved_ids, relevant_ids, k=5):
    """Compute standard retrieval metrics."""
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)

    # Precision@K
    precision = len(retrieved_set & relevant_set) / k

    # Recall@K
    recall = len(retrieved_set & relevant_set) / len(relevant_set) if relevant_set else 0

    # MRR
    mrr = 0
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_set:
            mrr = 1 / rank
            break

    return {'precision@k': precision, 'recall@k': recall, 'mrr': mrr}

```

**Common confusion:** High recall doesn't mean high quality answers. You might retrieve the relevant document but at position 10 where it gets ignored, or the relevant document might not contain a complete answer.

---

### Concept B: Answer-Ground Truth Comparison

**Definition:** Ground truth comparison measures how close the generated answer is to a known correct answer. This requires a test set with questions and expected answers, plus metrics that handle legitimate variation in phrasing.

**How it relates to Retrieval Scoring:** Retrieval scoring tells you if you found the right sources; ground truth comparison tells you if the final answer is correct. You can have perfect retrieval and still generate wrong answers (or vice versa, though rare).

**Key characteristics:**

- **Exact Match (EM):** Binary—does the answer exactly match? Useful for factoid questions.
- **F1 Score (Token Overlap):** Measures word-level overlap. Handles partial matches.
- **Semantic Similarity:** Embedding-based comparison. Catches paraphrased correct answers.
- **LLM-as-Judge:** Use an LLM to evaluate if the answer is correct given the expected answer.

**A concrete example:**

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_answer_metrics(generated, ground_truth):
    """Compare generated answer to ground truth."""
    # Exact match
    em = int(generated.strip().lower() == ground_truth.strip().lower())

    # Token F1
    gen_tokens = set(generated.lower().split())
    gt_tokens = set(ground_truth.lower().split())
    if len(gen_tokens) == 0 or len(gt_tokens) == 0:
        f1 = 0
    else:
        precision = len(gen_tokens & gt_tokens) / len(gen_tokens)
        recall = len(gen_tokens & gt_tokens) / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Semantic similarity
    embeddings = model.encode([generated, ground_truth])
    semantic_sim = np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )

    return {'exact_match': em, 'f1': f1, 'semantic_similarity': semantic_sim}

```

**Remember:** No single metric is perfect. Use multiple metrics and examine failure cases qualitatively to understand what's really happening.

---

### Concept C: Hallucination Detection and Fixing

**Definition:** Hallucination occurs when the LLM generates information not present in the retrieved context. Detection involves checking if each claim in the response is supported by the retrieved documents. Fixing involves prompt engineering, retrieval improvements, or post-generation filtering.

**Key characteristics:**

- **Claim extraction:** Break the response into individual factual claims
- **Grounding verification:** Check if each claim appears in or is supported by retrieved context
- **Grounding score:** Fraction of claims that are properly grounded (0-1)
- **Mitigation strategies:** Improve prompts, add citation requirements, post-filter ungrounded responses

**A concrete example:**

```python
def check_grounding(response, retrieved_chunks, threshold=0.7):
    """Check if response claims are grounded in retrieved context."""
    # Split response into sentences (simple claim extraction)
    claims = [s.strip() for s in response.split('.') if len(s.strip()) > 10]

    # Combine retrieved chunks
    context = ' '.join(retrieved_chunks)
    context_embedding = model.encode([context])[0]

    grounded_claims = []
    ungrounded_claims = []

    for claim in claims:
        claim_embedding = model.encode([claim])[0]
        similarity = np.dot(claim_embedding, context_embedding) / (
            np.linalg.norm(claim_embedding) * np.linalg.norm(context_embedding)
        )

        if similarity >= threshold:
            grounded_claims.append(claim)
        else:
            ungrounded_claims.append(claim)

    grounding_score = len(grounded_claims) / len(claims) if claims else 1.0

    return {
        'grounding_score': grounding_score,
        'grounded_claims': grounded_claims,
        'ungrounded_claims': ungrounded_claims
    }

```

---

### How These Concepts Work Together

Evaluation flows: Query → Retrieve → **Retrieval Scoring** → Generate → **Ground Truth Comparison** + **Hallucination Detection**. Poor retrieval causes downstream failures; good retrieval with poor generation also fails. Hallucination detection catches generation failures that slip past ground truth comparison (when the answer is plausible but invented).

---

## 3. Seeing It in Action: Worked Examples

### Example 1: Comprehensive Retrieval Evaluation

**Scenario:** A documentation search system needs to demonstrate 90% recall before launch. You need to build an evaluation pipeline that measures this rigorously.

**Our approach:** Create a test set with known relevant documents, implement multiple metrics, and generate a detailed report.

**Step-by-step solution:**

```python
# Step 1: Define test set structure
test_set = [
    {
        'query': 'How do I reset my password?',
        'relevant_doc_ids': ['doc_auth_001', 'doc_auth_003'],
        'expected_answer': 'Go to Settings > Security > Reset Password'
    },
    # ... more test cases
]

# Step 2: Run evaluation
def evaluate_retrieval_system(retrieval_fn, test_set, k_values=[1, 3, 5, 10]):
    """Comprehensive retrieval evaluation."""
    results = {f'recall@{k}': [] for k in k_values}
    results.update({f'precision@{k}': [] for k in k_values})
    results['mrr'] = []

    for item in test_set:
        retrieved = retrieval_fn(item['query'], k=max(k_values))
        retrieved_ids = [r['id'] for r in retrieved]
        relevant_ids = item['relevant_doc_ids']

        # Compute metrics for each K
        for k in k_values:
            metrics = compute_retrieval_metrics(retrieved_ids, relevant_ids, k)
            results[f'recall@{k}'].append(metrics['recall@k'])
            results[f'precision@{k}'].append(metrics['precision@k'])

        results['mrr'].append(metrics['mrr'])

    # Aggregate
    return {metric: sum(vals) / len(vals) for metric, vals in results.items()}

# Step 3: Generate report
metrics = evaluate_retrieval_system(search_fn, test_set)
print(f"Recall@5: {metrics['recall@5']:.2%}")
print(f"MRR: {metrics['mrr']:.3f}")

```

**Output:**

```
Recall@5: 87.5%
Recall@10: 94.2%
MRR: 0.782
Precision@5: 0.45

```

**What just happened:** We discovered that Recall@5 is below target (87.5% < 90%), but Recall@10 meets it. This suggests relevant documents are often in positions 6-10—we might need to retrieve more documents or improve ranking.

**Check your understanding:** Why might high recall with low precision still be acceptable for RAG systems?

---

### Example 2: LLM-as-Judge for Answer Evaluation

**Scenario:** Token-based metrics (F1) flag many correct answers as wrong because of paraphrasing. You need a more robust evaluation that understands semantic correctness.

**What's different:** Instead of string matching, we'll use an LLM to judge if the generated answer conveys the same information as the ground truth.

**Solution:**

```python
# Step 1: Define the judge prompt
JUDGE_PROMPT = """You are evaluating if an AI-generated answer is correct.

Question: {question}
Expected Answer: {expected}
Generated Answer: {generated}

Evaluate on these criteria:
1. Factual Accuracy: Does the generated answer contain correct facts?
2. Completeness: Does it cover the key information from the expected answer?
3. No Hallucination: Does it avoid adding false information?

Respond with:
CORRECT - if the answer is factually accurate and complete
PARTIAL - if partially correct but missing key information
INCORRECT - if factually wrong or contradicts the expected answer

Your judgment:"""

# Step 2: Implement evaluation
def llm_judge_answer(question, expected, generated, llm_client):
    prompt = JUDGE_PROMPT.format(
        question=question,
        expected=expected,
        generated=generated
    )

    response = llm_client.generate(prompt)

    # Parse judgment
    if 'CORRECT' in response.upper():
        return {'judgment': 'correct', 'score': 1.0}
    elif 'PARTIAL' in response.upper():
        return {'judgment': 'partial', 'score': 0.5}
    else:
        return {'judgment': 'incorrect', 'score': 0.0}

# Step 3: Aggregate across test set
def evaluate_with_llm_judge(test_set, rag_pipeline, llm_client):
    scores = []
    for item in test_set:
        generated = rag_pipeline.generate(item['query'])
        result = llm_judge_answer(
            item['query'],
            item['expected_answer'],
            generated,
            llm_client
        )
        scores.append(result['score'])

    return {
        'accuracy': sum(1 for s in scores if s == 1.0) / len(scores),
        'avg_score': sum(scores) / len(scores)
    }

```

**Key lesson:** LLM-as-judge can handle paraphrasing and semantic equivalence, but introduces its own biases. Use it alongside traditional metrics, not as a replacement.

---

### Example 3: Hallucination Detection and Mitigation

**Background:** A legal tech company found that their RAG system sometimes cited non-existent cases or invented legal precedents—extremely dangerous in legal contexts.

**The challenge:** Detect when the LLM generates claims not supported by retrieved documents, and implement safeguards.

**The approach:** Implement multi-level hallucination detection: (1) semantic grounding check, (2) citation verification, (3) confidence thresholding.

**Why this approach:** Single checks miss different types of hallucinations. Layered detection catches more issues while allowing legitimate responses through.

**The outcome:**

```python
class HallucinationDetector:
    def __init__(self, embedding_model, grounding_threshold=0.7):
        self.model = embedding_model
        self.threshold = grounding_threshold

    def detect(self, response, context_chunks):
        """Multi-level hallucination detection."""
        results = {
            'grounding_score': 0,
            'ungrounded_claims': [],
            'missing_citations': [],
            'risk_level': 'low'
        }

        # Level 1: Semantic grounding
        grounding = check_grounding(response, context_chunks, self.threshold)
        results['grounding_score'] = grounding['grounding_score']
        results['ungrounded_claims'] = grounding['ungrounded_claims']

        # Level 2: Citation verification (if citations present)
        citations = self._extract_citations(response)
        for citation in citations:
            if not self._verify_citation(citation, context_chunks):
                results['missing_citations'].append(citation)

        # Level 3: Risk assessment
        if results['grounding_score'] < 0.5 or len(results['missing_citations']) > 0:
            results['risk_level'] = 'high'
        elif results['grounding_score'] < 0.8:
            results['risk_level'] = 'medium'

        return results

    def mitigate(self, response, detection_results):
        """Apply mitigation strategies based on detection."""
        if detection_results['risk_level'] == 'high':
            return {
                'action': 'block',
                'message': 'Response could not be verified against sources.',
                'original_response': response
            }
        elif detection_results['risk_level'] == 'medium':
            return {
                'action': 'warn',
                'response': response,
                'warning': 'Some claims could not be fully verified.'
            }
        else:
            return {'action': 'pass', 'response': response}

```

**Caution:** Aggressive hallucination filtering can block legitimate responses. Tune thresholds based on your risk tolerance and monitor false positive rates.

---

## 4. Common Pitfalls: What Can Go Wrong and How to Avoid It

- **The Mistake:** Using only exact match for answer evaluation

**Why It's a Problem:** Correct answers phrased differently get marked wrong. "The capital is Paris" vs "Paris is the capital" both correct, but EM=0.
**The Right Approach:** Use multiple metrics: EM for factoid, F1 for overlap, semantic similarity for paraphrasing, LLM-judge for nuanced cases.
**Why This Works:** Different metrics catch different aspects of correctness. No single metric captures everything.

---

- **The Mistake:** Setting grounding thresholds too high

**Why It's a Problem:** Legitimate paraphrasing gets flagged as hallucination. If the context says "quarterly earnings increased" and the response says "profits rose each quarter," a strict check might flag it.
**The Right Approach:** Start with moderate thresholds (0.6-0.7), examine false positives, tune based on error analysis.
**Why This Works:** Thresholds should balance catching real hallucinations vs blocking valid responses. This requires empirical tuning.

---

- **The Mistake:** Ignoring retrieval quality when answer quality is low

**Why It's a Problem:** You might spend weeks improving the generation prompt when the real issue is that relevant documents aren't being retrieved.
**The Right Approach:** Always evaluate retrieval and generation separately. Diagnose which component is failing before fixing.
**Why This Works:** RAG is a pipeline—failures propagate. Fix the root cause, not the symptom.

**If you're stuck:** Sample 10 failure cases and manually examine what went wrong. Is it retrieval failure, generation failure, or evaluation failure?

---

## 5. Your Turn: Practice & Self-Assessment

### Practice Task

**The Challenge:** Build a complete evaluation suite that measures retrieval quality, answer correctness, and grounding for a RAG system.

**Specifications:**

- Create a test set with 15 question-answer pairs
- Implement Recall@5, MRR, F1, and semantic similarity metrics
- Add grounding detection with configurable threshold
- Generate a comprehensive evaluation report

**Hint:** Start with the retrieval metrics—they're foundational. If retrieval is failing, downstream metrics will be meaningless.

**Extension (optional):** Implement an LLM-as-judge evaluator and compare its judgments to your automated metrics.

---

### Check Your Understanding

1. **Explanation question:** Why is Recall@K more important than Precision@K for RAG systems specifically?
2. **Application question:** Your RAG system has high semantic similarity scores (0.85) but users report wrong answers. What might be happening?
3. **Error analysis:** A grounding detector flags 40% of responses as ungrounded, but manual review shows most are correct. What's likely wrong?
4. **Transfer question:** How would you adapt answer evaluation for a multilingual RAG system where questions and answers might be in different languages?

**Answers & Explanations:**

1. 
In RAG, the LLM can often extract the answer from multiple retrieved documents—we just need the right document to be present somewhere in the context. Recall measures this. Precision matters less because the LLM can ignore irrelevant retrieved documents (unlike a user scrolling through search results).

2. 
Semantic similarity measures if the answer is topically related, not if it's factually correct. A response about the wrong year's financial results might be semantically similar to the correct answer but factually wrong. Add factual accuracy checks, not just semantic similarity.

3. 
The grounding threshold is likely too high, or the semantic similarity method is too strict for paraphrasing. Lower the threshold and re-evaluate. Also check if the embedding model handles domain-specific terminology well.

4. 
Use multilingual embedding models (e.g., `paraphrase-multilingual-MiniLM-L12-v2`) for semantic similarity. For LLM-as-judge, use a multilingual LLM and consider translating to a common language for consistency. Stratify evaluation by language pair to catch language-specific issues.

---

### Self-Assessment Checklist

You've mastered this topic if you can:

- Implement and interpret retrieval metrics (Recall, Precision, MRR, NDCG)
- Compare generated answers to ground truth using multiple methods
- Design and implement hallucination detection systems
- Diagnose whether failures come from retrieval or generation
- Tune evaluation thresholds based on empirical error analysis
- Build end-to-end evaluation pipelines for RAG systems

**If you checked fewer than 5 boxes:** Focus on implementing the metrics hands-on—theory without practice doesn't stick.

---

## 6. Consolidation: Key Takeaways & Next Steps

### The Essential Ideas

- **Measure both retrieval and generation:** They fail differently and require different fixes.
- **Use multiple metrics:** No single metric captures answer quality—combine exact match, F1, semantic similarity, and LLM-as-judge.
- **Hallucination detection is essential:** In production systems, ungrounded claims can damage trust and cause real harm.

### Mental Model Check

By now, you should think of RAG evaluation as: a multi-stage diagnostic system that pinpoints where your pipeline fails—retrieval, generation, or grounding—so you can fix the right component.

### What You Can Now Do

You can build comprehensive evaluation systems that measure RAG quality across multiple dimensions, detect hallucinations before users see them, and guide data-driven optimization.

### Next Steps

**To deepen this knowledge:** Run your evaluation suite on different pipeline configurations and analyze which changes actually improve metrics.

**To build on this:** Learn about human evaluation protocols and how to combine automated and human judgments for maximum accuracy.

**Additional resources:** RAGAS library for standardized RAG evaluation, TruLens for LLM application monitoring.

---

## Quick Reference Card

Metric | Measures | Good For
Recall@K | % relevant docs in top-K | Retrieval coverage
MRR | Average rank of first relevant | Ranking quality
Exact Match | Binary string match | Factoid questions
Token F1 | Word overlap | Partial matches
Semantic Sim | Embedding similarity | Paraphrased answers
Grounding Score | % claims supported by context | Hallucination detection

---

**Questions or stuck?** Start with manual error analysis on 10 failures—it reveals more than aggregate metrics alone.

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