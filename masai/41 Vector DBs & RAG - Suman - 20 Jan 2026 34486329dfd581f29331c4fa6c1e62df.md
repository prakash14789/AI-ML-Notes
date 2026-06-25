# 41. Vector DBs & RAG - Suman - 20 Jan 2026

# Vector Databases & Retrieval-Augmented Generation (RAG)

## [PPT File](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/7c161333-ee34-4935-bd49-e777c5a7ffc0/pAAXREVdJQUpkKjm.pdf)

**Prerequisites:** Basic Python programming, familiarity with embeddings (word2vec or sentence transformers), understanding of how LLMs generate text.

**Time to complete:** 35-45 minutes

**What you'll be able to do:**

- Explain what vector databases are and why they're essential for RAG systems
- Implement text chunking strategies for optimal retrieval
- Build a functional FAQ bot using FAISS or Chroma
- Evaluate retrieval quality and grounding in RAG outputs

---

## 1. Introduction: What are Vector Databases and Why Should You Care?

### Core Definition

A **vector database** is a specialized database designed to store, index, and query high-dimensional vectors (embeddings). Unlike traditional databases that match exact values, vector databases find items based on semantic similarity—how close two pieces of content are in meaning, not just text.

**Retrieval-Augmented Generation (RAG)** combines the generative power of LLMs with external knowledge retrieval. Instead of relying solely on what the model learned during training, RAG fetches relevant documents from a knowledge base and uses them to ground the response.

### A Simple Analogy

Think of a vector database like a librarian with perfect memory and understanding. When you ask about "machine learning optimization," they don't just look for books with those exact words—they understand what you mean and bring you books about gradient descent, hyperparameter tuning, and neural network training, even if those exact words aren't in your query.

**Limitation:** Unlike a librarian who understands context deeply, vector similarity can sometimes retrieve documents that share similar words but different meanings (semantic overlap without true relevance).

### Why This Matters to You

**Problem it solves:** LLMs have knowledge cutoffs and can hallucinate facts. RAG systems ground responses in your actual documents—company policies, product manuals, research papers—ensuring accuracy and recency.

**What you'll gain:**

- Build chatbots that answer from your specific documentation
- Create search systems that understand intent, not just keywords
- Reduce LLM hallucinations by providing factual context

**Real-world context:** ChatGPT plugins, Notion AI, and enterprise knowledge bases all use RAG to connect LLMs with proprietary data.

---

## 2. The Foundation: Core Concepts Explained

### Concept A: Text Chunking

**Definition:** Text chunking is the process of splitting large documents into smaller, semantically meaningful pieces that can be independently embedded and retrieved. The chunk size affects both retrieval precision and the context available to the LLM.

**Key characteristics:**

- **Chunk size matters:** Too small (50 tokens) loses context; too large (2000 tokens) dilutes relevance
- **Overlap prevents boundary issues:** 10-20% overlap ensures ideas spanning chunk boundaries aren't lost
- **Semantic chunking > fixed-size:** Splitting at paragraph or section boundaries preserves meaning better than arbitrary character counts

**A concrete example:**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)

chunks = splitter.split_text(document_text)

```

**Common confusion:** Beginners often use very small chunks thinking "more precise retrieval." In reality, small chunks lack the context needed for the LLM to generate coherent answers.

---

### Concept B: Vector Stores (FAISS & Chroma)

**Definition:** Vector stores are the engines that index embeddings and perform similarity search. FAISS (Facebook AI Similarity Search) is optimized for speed with millions of vectors; Chroma is developer-friendly with built-in persistence and metadata filtering.

**How it relates to Text Chunking:** Each chunk gets converted to an embedding vector, then stored in the vector database. At query time, your question becomes a vector, and the database finds the closest chunk vectors.

**Key characteristics:**

- **FAISS:** Blazing fast, supports billion-scale indices, requires more setup
- **Chroma:** Easy to start, built-in persistence, great for prototyping and medium-scale apps
- **Both support:** Approximate nearest neighbor (ANN) search for speed, exact search for precision

**A concrete example:**

```python
# Chroma example
import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
client = chromadb.Client()
collection = client.create_collection("faq_docs", embedding_function=ef)

collection.add(
    documents=["How do I reset my password?", "What are the pricing tiers?"],
    ids=["doc1", "doc2"]
)

results = collection.query(query_texts=["forgot password"], n_results=1)

```

**Remember:** FAISS requires you to manage embeddings separately, while Chroma handles embedding generation if you provide an embedding function.

---

### How Chunking and Vector Stores Work Together

The pipeline flows: Raw Document → Chunker (splits into pieces) → Embedding Model (converts to vectors) → Vector Store (indexes for search). At query time: User Question → Embedding → Vector Store Query → Top-K Chunks → LLM generates answer using chunks as context.

---

## 3. Seeing It in Action: Worked Examples

### Example 1: Building an FAQ Bot

**Scenario:** You have 50 FAQ documents and want to build a bot that retrieves the most relevant answer when users ask questions.

**Our approach:** Chunk the FAQs (each Q&A pair is naturally one chunk), embed them with a sentence transformer, store in Chroma, and retrieve the top-3 matches for each user query.

**Step-by-step solution:**

```python
# Step 1: Prepare FAQ data
faqs = [
    {"q": "How do I reset my password?", "a": "Go to Settings > Security > Reset Password."},
    {"q": "What payment methods are accepted?", "a": "We accept Visa, Mastercard, and PayPal."},
    {"q": "How do I cancel my subscription?", "a": "Navigate to Account > Subscription > Cancel."}
]

# Step 2: Create combined documents for embedding
documents = [f"Q: {faq['q']}\nA: {faq['a']}" for faq in faqs]

# Step 3: Store in Chroma
import chromadb
client = chromadb.Client()
collection = client.create_collection("faq_bot")
collection.add(documents=documents, ids=[f"faq_{i}" for i in range(len(faqs))])

# Step 4: Query
user_question = "how can I change my password"
results = collection.query(query_texts=[user_question], n_results=1)
print(results['documents'][0][0])  # Returns the password reset FAQ

```

**Output:**

```
Q: How do I reset my password?
A: Go to Settings > Security > Reset Password.

```

**What just happened:** The user asked about "changing password" (not exact match to "reset"), but semantic similarity found the right FAQ because the embeddings captured the intent.

**Check your understanding:** Why did we combine Q and A into one document instead of embedding them separately?

---

### Example 2: Retrieval with FAISS for Scale

**Scenario:** You have 100,000 product descriptions and need sub-second retrieval for a recommendation system.

**What's different:** Chroma is convenient but FAISS is designed for this scale. We'll use FAISS with pre-computed embeddings.

**Solution:**

```python
# Step 1: Generate embeddings
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
product_descriptions = ["Wireless Bluetooth headphones...", "..."]  # 100k items
embeddings = model.encode(product_descriptions)

# Step 2: Create FAISS index
dimension = embeddings.shape[1]  # 384 for MiniLM
index = faiss.IndexFlatL2(dimension)  # L2 distance
index.add(np.array(embeddings).astype('float32'))

# Step 3: Query
query = "noise cancelling earbuds"
query_embedding = model.encode([query])
distances, indices = index.search(np.array(query_embedding).astype('float32'), k=5)

```

**Key lesson:** FAISS separates embedding from storage, giving you control but requiring more code. For production at scale, FAISS with IVF (inverted file) indexing can search billions of vectors in milliseconds.

---

### Example 3: Grounding Evaluation for RAG

**Background:** A legal tech company built a RAG system for contract analysis. They needed to verify that LLM responses were actually grounded in retrieved documents, not hallucinated.

**The challenge:** How do you measure if the generated answer faithfully represents the retrieved context?

**The approach:** They implemented a grounding score: for each claim in the LLM output, check if supporting evidence exists in the retrieved chunks. Scores range from 0 (no grounding) to 1 (fully grounded).

**Why this approach:** Automated metrics like ROUGE or BLEU measure text overlap but miss semantic grounding. A claim can be paraphrased (low ROUGE) but still grounded (high semantic match).

**The outcome:** They found 15% of responses contained claims not present in retrieved documents. By adding a grounding check before displaying answers, they reduced user-reported inaccuracies by 60%.

**Caution:** Grounding evaluation requires careful prompt engineering for the evaluator LLM, or it may incorrectly flag paraphrased-but-grounded statements.

---

## 4. Common Pitfalls: What Can Go Wrong and How to Avoid It

- **The Mistake:** Using chunk sizes that are too small (under 100 tokens)

**Why It's a Problem:** Retrieved chunks lack context. The LLM sees fragments like "...the answer is 42..." without knowing what question was asked.
**The Right Approach:** Use 300-800 token chunks with 10-20% overlap. Test retrieval quality with real queries.
**Why This Works:** Larger chunks preserve the semantic unit (a complete thought, paragraph, or section).

---

- **The Mistake:** Not filtering by metadata before vector search

**Why It's a Problem:** A user asking about "2024 pricing" might retrieve 2022 pricing documents because they're semantically similar.
**The Right Approach:** Store metadata (date, category, source) and filter before or after vector search.
**Why This Works:** Combining semantic search with structured filters gives you relevance AND precision.

---

- **The Mistake:** Assuming retrieval success means RAG success

**Why It's a Problem:** You can retrieve the perfect document, but if the LLM ignores it or misinterprets it, the answer is still wrong.
**The Right Approach:** Evaluate the full pipeline: retrieval recall, grounding score, and answer correctness.
**Why This Works:** End-to-end evaluation catches failures at every stage, not just retrieval.

**If you're stuck:** Revisit the chunking strategy first—80% of RAG quality issues trace back to how documents were chunked.

---

## 5. Your Turn: Practice & Self-Assessment

### Practice Task (Estimated: 15-20 minutes)

**The Challenge:** Build a documentation Q&A bot using Chroma that can answer questions about a technical topic of your choice (Python docs, a framework, your own notes).

**Specifications:**

- Collect at least 10 documentation pages or FAQ items
- Implement chunking with overlap
- Store in Chroma and implement a query function
- Test with 5 questions and evaluate if retrieved chunks are relevant

**Hint:** Start by manually checking if the retrieved chunks contain the answer before worrying about LLM integration. The retrieval quality is your foundation.

**Extension (optional):** Add metadata filtering so users can scope queries to specific sections or dates.

---

### Check Your Understanding

1. **Explanation question:** Why is 10-20% chunk overlap recommended, and what problem does it solve?
2. **Application question:** You're building a RAG system for a law firm with documents from 2010-2024. How would you handle a query like "What's the current policy on remote work?"
3. **Error analysis:** A RAG system retrieves relevant documents but the LLM still gives wrong answers. What could be causing this, and how would you debug it?
4. **Transfer question:** How would you adapt a RAG system designed for English documents to support multilingual queries?

**Answers & Explanations:**

1. 
Overlap ensures that if an important concept spans a chunk boundary, it appears in at least one complete chunk. Without overlap, you might split "The answer to question 5 is 42" across two chunks, making neither useful.

2. 
Store document dates as metadata. Filter to only search documents from the last 1-2 years for "current" queries. Alternatively, retrieve from all years but include dates in the context so the LLM can reason about recency.

3. 
Possible causes: (a) Retrieved chunks are related but don't contain the answer, (b) LLM context window is overloaded with too many chunks, (c) Prompt doesn't instruct LLM to use the context. Debug by examining retrieved chunks manually and testing with explicit "Answer based only on the following context" prompts.

4. 
Use a multilingual embedding model (like `paraphrase-multilingual-MiniLM-L12-v2`), ensure documents are stored in their original language, and consider language metadata for filtering. The embedding model handles cross-lingual similarity.

---

### Self-Assessment Checklist

You've mastered this topic if you can:

- Explain vector databases to a colleague without looking at notes
- Choose appropriate chunk sizes for different document types
- Implement a basic RAG pipeline with Chroma or FAISS
- Identify and fix common retrieval quality issues
- Evaluate both retrieval and grounding quality in a RAG system
- Adapt chunking strategy based on document structure

**If you checked fewer than 5 boxes:** Revisit the worked examples and try implementing them yourself—hands-on practice solidifies understanding.

---

## 6. Consolidation: Key Takeaways & Next Steps

### The Essential Ideas

- **Vector databases enable semantic search:** They find conceptually similar content, not just keyword matches, making them essential for connecting LLMs to your data.
- **Chunking quality determines retrieval quality:** Thoughtful chunking with appropriate size and overlap is the foundation of any RAG system.
- **Evaluate the full pipeline:** Retrieval success doesn't guarantee answer quality—measure grounding and correctness too.

### Mental Model Check

By now, you should think of RAG as: a pipeline where your documents become searchable vectors, user questions find similar vectors, and retrieved text grounds the LLM's response in factual context.

### What You Can Now Do

You can build a functional RAG system that retrieves relevant documents and provides grounded answers. This connects LLMs to any knowledge base—company docs, research papers, or product catalogs.

### Next Steps

**To deepen this knowledge:** Experiment with different embedding models and chunking strategies on the same document set—compare retrieval quality.

**To build on this:** Learn about hybrid search (combining vector and keyword search) and re-ranking for even better retrieval.

**Additional resources:** LangChain documentation for RAG patterns, Chroma's getting started guide for hands-on practice.

---

## Quick Reference Card

Component | Purpose | Key Choice
Chunker | Split documents | 300-800 tokens, 10-20% overlap
Embedding Model | Convert to vectors | MiniLM for speed, OpenAI for quality
Vector Store | Index and search | Chroma for dev, FAISS for scale
Retriever | Fetch top-K | K=3-5 for most use cases
Grounding Check | Verify accuracy | Compare claims to retrieved context

---

**Questions or stuck?** Start with the Chroma documentation—it has excellent tutorials for RAG beginners.

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