# 80. RAG at Scale with Spark - Suman - 20 May 2026

# RAG at Scale with Apache Spark

## PPT Link: [Click Here](https://drive.google.com/file/d/1fFqmA_PNlXrmuQNWpGzKfWgXn-QHSYT1/view?usp=sharing)

## 1. What You'll Learn in This Section

In this lesson, you'll learn to:

- Explain how Retrieval-Augmented Generation (RAG) solves the core limitations of large language models.
- Describe how Apache Spark distributes embedding generation across a cluster at big-data scale.
- Apply Locality-Sensitive Hashing (LSH) to reduce search space and speed up similarity lookups.
- Combine keyword and semantic search in a hybrid retrieval pipeline and improve result quality with re-ranking.

---

## 2. Detailed Explanation

### Why RAG is needed

**RAG (Retrieval-Augmented Generation)** is a technique that gives a large language model (LLM) access to up-to-date, external knowledge at query time — without retraining the model.

**Why it matters**

LLMs have three core problems that make them unreliable on their own:

1. **Hallucination** — an LLM always generates a response, even when the correct answer was absent from its training data. It produces plausible-sounding but factually wrong output because it never knows when it does not know the answer.
2. **No access to private data** — internal documents, company policies, and private files are never included in public LLM training data. Knowledge hidden in those files is unavailable to a base LLM.
3. **Expensive fine-tuning** — updating a model's knowledge by fine-tuning is computationally very expensive and not a scalable way to keep it current.

All three problems share the same root: the LLM's knowledge is static, fixed at training time. RAG solves this by giving the model access to dynamic, real-time knowledge without retraining.

**Walkthrough**

The cost comparison makes the trade-off clear: fine-tuning or retraining an LLM to incorporate updated knowledge is far more expensive than using RAG, which simply retrieves up-to-date information and passes it to the existing model.

RAG combines two components:

- The **retriever** fetches relevant documents from a knowledge base based on the user's query.
- The **LLM** generates the final answer using the retrieved documents as context.

The LLM still generates the response; it does so using retrieved context rather than relying solely on its training weights. This enables real-time knowledge access.

**Benefits of RAG:**

- Reduces hallucination by grounding responses in retrieved documents.
- Enables use of external and private knowledge.
- Keeps information up to date at query time.
- Lower cost than fine-tuning or retraining.
- Better explainability — source documents used for the answer can be traced.

A practical example: a query like "Who is the Chief Minister of West Bengal?" may have a different or unavailable answer in an LLM's training data if the information post-dates training. With RAG, the retriever fetches current news or government sources, and the LLM generates a correct, up-to-date answer from those documents.

**Common mistakes**

- Assuming RAG eliminates hallucination entirely — it reduces hallucination by grounding responses, but the retriever must find genuinely relevant documents for the grounding to help.
- Confusing RAG with fine-tuning — fine-tuning updates the model's weights permanently and is expensive; RAG retrieves knowledge dynamically at query time without touching the model.

---

### The RAG workflow

The end-to-end RAG flow applies both when storing documents and when processing a user query.

**Why it matters**

Understanding the full pipeline helps you see where each component fits and what can go wrong at each stage. A gap anywhere — bad chunking, slow embedding, or a poor search step — degrades the final answer quality.

**Walkthrough**

```
Document arrivesChunkingEmbedding computationVector database storageUser queryQuery embeddingSimilarity searchTop documents retrievedLLM generates answer
```

Step by step:

1. A document arrives from a data source (or a user query arrives from the interface).
2. **Chunking** — the document is divided into smaller parts (chunks).
3. **Embedding computation** — each chunk is converted into a numeric vector (embedding).
4. **Storage** — embeddings are stored in a vector database (for documents) or used directly (for queries).
5. **Search** — the query embedding is compared against stored embeddings to find the most similar ones.
6. **Retrieval** — the top matching documents are retrieved.
7. **LLM generation** — retrieved documents are passed to the LLM as context, and the LLM generates the final answer.

A complete RAG system includes these components:

Component | Role
Data source | Origin of the documents
Document store | Holds raw or processed documents
Chunking module | Divides documents into smaller segments
Embedding model | Converts text chunks into numeric vectors
Vector database | Stores and indexes embeddings for fast search
Retriever | Queries the vector database based on input
Re-ranker | Refines retrieved results by relevance score
LLM generator | Produces the final answer using retrieved context
User interface | Accepts the user query and displays the response

**Common mistakes**

- Skipping chunking and embedding an entire document as one unit — this produces a coarse embedding that loses the nuance of individual sections.
- Using the query raw (without embedding it) for vector search — semantic search only works when both the query and stored content are in the same embedding space.

---

### Chunking strategies

**Chunking** is the process of dividing a document into smaller pieces before embedding. Think of it as slicing a long book into chapters, then paragraphs, so each piece can be understood on its own.

**Why it matters**

Embedding a 100-page document as one vector loses detail. Chunking ensures each embedding represents a focused, meaningful unit of information. The right chunking strategy directly affects retrieval quality.

**Walkthrough**

Three strategies exist:

1. 
**Fixed-size chunking** — divides the document into equal-size chunks by character or token count. Simple and predictable.

2. 
**Recursive chunking** — divides first into paragraphs. If a paragraph is still too large, it divides further into sentences, then words. Size is reduced recursively until the target chunk size is met.

3. 
**Semantic chunking** — divides based on meaning and contextual similarity rather than size. Words and sentences that relate to the same concept are grouped together. For example, a chunk labelled "Spark overview" groups content about Spark architecture and applications because they share semantic meaning, even if those exact words do not appear in the source text.

**Common mistakes**

- Using chunks that are too large — they carry irrelevant noise and produce lower-quality embeddings.
- Using chunks that are too small — each embedding loses enough context that the semantic search returns poor matches.

---

### Embeddings

An **embedding** is the numeric vector representation of a piece of text — for example, "Spark is scalable" becomes an n-dimensional numeric vector.

**Why it matters**

Text cannot be compared mathematically as raw strings. By converting text to vectors, search becomes semantic: you find content with similar *meaning*, not just the same exact keyword.

**Walkthrough**

Key properties of embeddings:

- Embeddings with similar meaning are positioned close together in vector space. For example, the embedding of "car" and the embedding of "automobile" will be nearby vectors because the words are semantically similar.
- When searching a vector database, the search is **semantic** — it finds content with similar meaning, not just the same keyword.

Common embedding dimension ranges are **384**, **768**, and **1024** dimensions. Higher dimension means richer representation but also higher computational cost.

**Embedding models** compute embeddings from text. Models come from various providers, including open-source options and proprietary models from companies such as OpenAI and Microsoft.

**Common mistakes**

- Treating embeddings as exact keyword lookup — embeddings find semantically similar content, not just lexically identical text.
- Picking the highest-dimension model by default — higher dimensions increase computational cost; choose based on your accuracy and latency requirements.

---

### Vector databases and similarity metrics

A **vector database** stores embeddings and provides efficient mechanisms to search them by similarity. When embeddings are stored, they are organised so that similarity search is fast.

**Why it matters**

Raw vectors are just numbers; a vector database turns them into a queryable index. The similarity metric you choose determines what "close" means — Euclidean distance, cosine similarity, or dot product — and directly affects retrieval quality.

**Walkthrough**

To compare two embeddings (one from a query, one stored in the database), three common distance and similarity measures are used:

Metric | What it measures | Best for
Euclidean distance | Straight-line distance between two vectors in n-dimensional space | Geometric/spatial problems
Cosine similarity | Angle between two vectors; high similarity means vectors point in similar directions regardless of magnitude | Text and NLP (magnitude-independent)
Dot product | Sum of element-wise products; related to cosine similarity but sensitive to vector magnitude | When vector magnitude matters

Cosine similarity is the most common choice for text embeddings because it measures directional similarity — whether two vectors point in the same direction — rather than absolute distance. This works well for text because the magnitude of a word vector is less important than its direction in semantic space.

**Common mistakes**

- Assuming all similarity metrics produce identical rankings — the choice of metric can significantly change which documents are retrieved.
- Storing embeddings without a vector database — linear search (comparing against every embedding) becomes prohibitively slow at scale; a vector database uses efficient indexing to speed up queries.

---

### Scale challenges and Apache Spark

At small scale (thousands of documents on a single machine), straightforward approaches work fine. At large scale (millions or billions of documents), those approaches break down.

**Why it matters**

When your data volume crosses into "big data" territory, single-machine processing cannot keep up. The solution is distributed processing — splitting the work across multiple machines. Apache Spark is the standard framework for this.

**Walkthrough**

Scale challenges at large document volumes:

Challenge | Effect at scale
Embedding generation cost | Becomes prohibitively expensive on a single machine
Query latency | Time between firing a query and receiving a response grows unacceptably large
Storage | Volume of embeddings overwhelms single-machine storage
Memory consumption | Cannot hold all required data in RAM on one node
Multi-user traffic | Many concurrent users amplify all of the above

**Apache Spark** is a distributed data processing framework for handling big data. It executes computation across multiple nodes simultaneously.

Key features:

1. **In-memory computation** — intermediate datasets are stored in RAM rather than written to disk between computation steps. This avoids the repeated disk I/O that characterises Hadoop's MapReduce approach, making Spark significantly faster for iterative algorithms.
2. **Parallel processing** — work is distributed across multiple nodes that execute simultaneously.
3. **Fault tolerance** — Spark uses three mechanisms to recover from node failures (covered below).
4. **Distributed execution** — operations are carried out across a cluster of nodes.
5. **Horizontal scalability** — nodes can be added as data volume grows; removing nodes reduces the cluster when requirements shrink.

**Spark vs Hadoop:** Hadoop writes every intermediate computation result to disk, causing heavy disk I/O. Spark stores intermediate results in memory and only writes to disk when necessary, which is faster when the data volume fits in available RAM. When datasets are so large that they cannot fit in memory, Hadoop's disk-based approach is necessary. PySpark (Spark's Python interface) is widely adopted because it integrates well with Python-based machine learning and LLM workflows.

**Common mistakes**

- Assuming Spark is always better than Hadoop — when the dataset cannot fit in RAM, Hadoop's disk-based approach is the appropriate choice.
- Starting with a single-machine approach and hoping to scale it later — plan for distributed processing from the start when you know data volumes will be large.

---

### Spark fault tolerance

Spark achieves fault tolerance through three mechanisms that allow it to recover from node failures without losing data.

**Why it matters**

In a cluster of many machines, individual nodes fail regularly. A data processing framework that cannot survive node failures is unreliable for production workloads.

**Walkthrough**

1. 
**RDD lineage** — Spark records the complete sequence of transformations used to produce a dataset rather than storing the intermediate data itself. If data is lost, Spark replays those exact transformations from the original source to reconstruct it. For example, if a value was produced by squaring x, dividing by 3, and taking the log, and that result is lost, Spark re-executes those exact steps to regenerate it.

2. 
**Data replication** — data is stored on multiple nodes so that if one node fails, the data can be retrieved from another.

3. 
**Lazy evaluation** — Spark does not execute a transformation immediately when it is written. Transformations are queued and executed only when an action (such as `.collect()`) is called. This allows Spark to store the full lineage of transformations and reconstruct data on demand. Until `.collect()` is called, no actual computation occurs.

**Common mistakes**

- Calling `.collect()` on a very large dataset — `.collect()` brings all data to the driver node. On a billion-record dataset, this exhausts driver memory. Use actions like `.count()` or `.take(n)` for exploratory checks instead.
- Assuming lazy evaluation means nothing is wrong if no error appears immediately — errors surface only at the action call, not at the transformation definition.

---

### Spark architecture and ecosystem

A Spark cluster has three roles, and understanding them helps you reason about where computation actually happens.

**Why it matters**

Knowing the architecture helps a data engineer configure cluster resources correctly, debug performance bottlenecks, and understand why adding more executor nodes speeds up a job.

**Walkthrough**

Role | Responsibility
Cluster manager | Allocates resources across the cluster; manages the overall cluster
Driver node | Controls and coordinates the execution of a Spark job
Executor nodes | Perform the actual computation; multiple executors run in parallel

Actual computation is done by executor nodes. Multiple executor nodes work simultaneously, each handling a partition of the data.

The Spark ecosystem includes several components:

- **Spark Core** — the foundational execution engine.
- **Spark SQL** — structured data querying.
- **Spark Streaming** — processing of streaming (real-time) data.

**PySpark** is the Python library for Spark, enabling Python-based code to run distributed Spark jobs. Scala can also be used with Spark.

**Common mistakes**

- Running all computation on the driver node by pulling data to the driver with `.collect()` — keep computation on the executor nodes and only collect final, small results.
- Confusing PySpark with a completely different tool — PySpark is simply the Python API for Spark; all the distributed Spark concepts apply unchanged.

---

### Distributed embedding generation with Spark

When documents number in the millions, embedding generation moves from a sequential single-machine process to a distributed one using Spark. The fundamental steps remain the same; what changes is that computation happens in parallel across executor nodes.

**Why it matters**

Generating embeddings for one million documents sequentially on a single machine would take far too long. Distributing the work across a Spark cluster cuts the time proportionally to the number of executor nodes.

**Walkthrough**

Steps for distributed embedding generation:

1. Read documents.
2. Convert to a Spark DataFrame.
3. Distribute the DataFrame across executor nodes (partitions).
4. Each executor independently computes embeddings for its partition.
5. Gather embeddings across all executors.
6. Store embeddings in the vector database.

Each partition is processed independently. This achieves **horizontal scaling**: as document volumes grow, more executor nodes can be added.

The following PySpark code illustrates this pattern:

```python
from pyspark.sql import SparkSession

# Create a Spark session — entry point for all Spark operations
spark = SparkSession.builder.appName("EmbeddingGeneration").getOrCreate()

# Read source data
df = spark.read.text("data/documents")

# Convert to RDD; this is lazy — no computation happens yet
rdd = df.rdd

# Define the embedding computation — still lazy, just queued
embeddings_rdd = rdd.map(lambda row: compute_embedding(row))

# .collect() triggers execution across all executor nodes in parallel
embeddings = embeddings_rdd.collect()

for embedding in embeddings:
    store_in_vector_db(embedding)

```

Key points:

- `SparkSession` is the entry point for Spark operations.
- Transformations like `.map()` are **lazy** — defined but not executed until an action is called.
- `.collect()` is the action that triggers execution; all executors compute embeddings for their partitions in parallel.

Data preparation for RAG follows the **ETL** (Extract, Transform, Load) pattern:

- **Extract** — collect documents from sources.
- **Transform** — clean the data, remove noise, perform chunking, compute embeddings.
- **Load** — store embeddings and associated metadata in the vector database.

At scale, all three phases benefit from Spark's distributed execution.

**Common mistakes**

- Loading the embedding model outside the worker function — the model must be loaded inside the function passed to `.map()` so it is available on each executor node, not just the driver.
- Forgetting that `.map()` is lazy — writing a `.map()` call and wondering why nothing ran is a classic beginner mistake; you need an action like `.collect()` to trigger execution.

---

### Approximate nearest-neighbor (ANN) search

**Exact nearest-neighbor search** compares a query embedding against every stored embedding and sorts all distances. With 1 billion stored embeddings, that means 1 billion comparisons per query — unacceptably slow at scale.

**ANN (Approximate Nearest-Neighbor) search** sacrifices a small amount of accuracy to achieve dramatic speed improvements. It does not guarantee the exact closest vector but returns one that is very close.

**Why it matters**

At production scale, exact nearest-neighbor search is not viable. ANN enables search that is fast enough for real-time responses while remaining accurate enough for practical use.

**Walkthrough**

Approach | Accuracy | Speed | Scalability
Exact nearest-neighbor | Perfect | Very slow at large scale | Poor
ANN | Almost exact (minor error possible) | Very fast | Good

Several ANN algorithms exist. The choice depends on dataset size, available memory, desired speed, and acceptable accuracy. There is no single best algorithm; the right choice is context-dependent.

**Common mistakes**

- Rejecting ANN because it is "approximate" — in practice, the small accuracy loss is acceptable and the speed gain is essential at scale.
- Treating all ANN algorithms as interchangeable — different algorithms have different memory, speed, and accuracy profiles; benchmark before choosing.

---

### Locality-Sensitive Hashing (LSH)

**LSH (Locality-Sensitive Hashing)** is one technique for computing approximate nearest neighbors. It applies hashing to reduce the search space dramatically — similar to how a mod-10 hash function sends you directly to the right bin rather than scanning every stored number.

**Why it matters**

LSH converts the nearest-neighbor search problem into a simple bucket lookup. Instead of comparing a query against every stored vector, you compute a bucket ID and search only that one bucket.

**Walkthrough**

Traditional hashing maps data items into buckets using a hash function. For example, a mod-10 hash maps any integer to one of ten buckets (0–9) based on the remainder. When searching for a value, only the relevant bucket needs to be inspected.

LSH applies the same principle to high-dimensional embedding vectors:

1. A hash function is applied to each vector. For each dimension, the function produces a binary bit:

If the value in that dimension is below a threshold (e.g., 0.5) → bit = 0.
If the value is at or above the threshold → bit = 1.

2. The resulting binary string (one bit per dimension) becomes the **bucket ID** for that vector.
3. All vectors sharing the same bucket ID are stored in the same bucket.
4. When a query arrives, its bucket ID is computed with the same hash function. Only the vectors in that one bucket need to be compared.

**Bucket count:** for an n-dimensional embedding, each dimension produces 2 possible bit values, so total buckets = 2^n.

Dimensions | Buckets
2 | 4
3 | 8
5 | 32
n | 2^n

For 768-dimensional vectors, the reduction in search space is enormous.

**2D example:** for 2-dimensional vectors where x and y range from 0 to 1, a threshold of 0.5 divides the space into four quadrants:

Bucket ID | Condition
00 | x < 0.5 and y < 0.5
10 | x ≥ 0.5 and y < 0.5
01 | x < 0.5 and y ≥ 0.5
11 | x ≥ 0.5 and y ≥ 0.5

A query vector (0.4, 0.8) computes to bucket ID 01 (0.4 < 0.5 → bit 0; 0.8 ≥ 0.5 → bit 1). The search is restricted to that one bucket.

**Why LSH is approximate — the boundary problem:** vectors near the threshold boundary can land in different buckets even though they are physically very close. A query vector (0.49, 0.49) hashes to bucket 00; a stored vector (0.50, 0.50) hashes to bucket 11. These two vectors are extremely close, but because one falls just below the threshold and the other just meets it, they land in different buckets. The search in bucket 00 will never find the vector in bucket 11, even though it is the actual nearest neighbor.

**Common mistakes**

- Expecting LSH to always return the exact nearest neighbor — approximation error occurs at bucket boundaries; plan for this in your application.
- Computing bucket IDs inconsistently — the query and stored vectors must use the same hash function and the same threshold; any mismatch sends the query to the wrong bucket.

---

### Hybrid search

**Hybrid search** combines two retrieval mechanisms to improve result quality: semantic (vector) search and keyword-based search.

**Why it matters**

Neither approach alone is perfect. Keyword search misses semantically related content that uses different words. Semantic search can miss exact-term matches that keyword search would catch. Combining both produces better recall and precision.

**Walkthrough**

The two components:

1. 
**Semantic (vector) search** — converts the query to an embedding and searches for semantically similar vectors using ANN techniques. Finds results with similar meaning, not just identical keywords.

2. 
**Keyword-based search (BM25)** — retrieves documents based on exact keyword matches. Used in traditional search engines and information retrieval systems such as Elasticsearch.

Combining the two produces a **hybrid score** that captures both exact-term matches and semantically related content.

Example: a search for "Docker" via keyword search retrieves documents that mention Docker explicitly. Vector search may also retrieve documents about containers (a semantically related concept), since Docker is a container platform and the embeddings of "Docker" and "container" are close in vector space.

The hybrid search pipeline:

1. Run keyword-based search (e.g., BM25) → get keyword-match scores.
2. Run vector/semantic search (e.g., using LSH) → get semantic similarity scores.
3. Merge the two result sets.
4. Re-rank the merged results.
5. Pass the top results as context to the LLM for generation.

**Common mistakes**

- Using only semantic search and assuming it is sufficient — exact-term matching is important for domain-specific terminology that may not appear in training data for the embedding model.
- Skipping the merge and re-ranking step — raw hybrid results contain duplicates and inconsistent scoring; merging and re-ranking is what makes hybrid search coherent.

---

### BM25 keyword search

**BM25 (Best Matching 25)** is the standard keyword-based ranking algorithm used in search engines. It computes a relevance score for each document with respect to a query, using three factors.

**Why it matters**

BM25 does not treat all matching words equally. It gives more weight to rare, distinctive terms (high IDF) and less weight to very common words — producing more relevant rankings than a simple word-count approach.

**Walkthrough**

The three factors BM25 uses:

- **Term frequency (TF)** — how many times a query term appears in the document.
- **Inverse document frequency (IDF)** — how rare a term is across all documents.
- IDF is computed as log(N / df_t), where N is the total document count and df_t is the count of documents containing the term.
- **Document length** — longer documents are normalised to avoid a bias toward length.

**Why "inverse" document frequency:** IDF is high for terms that appear in few documents (rare terms) and low for terms that appear in many documents (common terms). By placing df_t in the denominator, the formula gives less weight to very frequent terms across the corpus and more weight to distinctive, rare terms. Matching on an uncommon word carries more significance than matching on a ubiquitous word.

Documents are ranked by their BM25 score in descending order to produce the final keyword-search result list.

**Common mistakes**

- Treating BM25 as a simple word-count match — BM25 accounts for term rarity and document length, making it far more sophisticated than counting occurrences.
- Using keyword search alone for semantic queries — BM25 will miss documents that describe the same concept with different vocabulary; pair it with vector search for full coverage.

---

### Re-ranking

**Re-ranking** takes the initial retrieval results and reorders them by a deeper relevance score, improving the quality of the final context passed to the LLM.

**Why it matters**

Initial retrieval (whether from semantic search, keyword search, or hybrid search) may include noisy or weakly relevant documents. Re-ranking adds a quality gate before the LLM sees the results.

**Walkthrough**

The re-ranking process:

1. Retrieve the top-k documents using the hybrid search pipeline.
2. For each retrieved document, compute a relevance score between the query and that document.
3. Reorder the retrieved documents by relevance score (descending).
4. Pass the reordered top results to the LLM.

**Cross-encoder re-ranking** is one technique for computing relevance scores. It uses a deep neural network that takes the query and a document *together* as input and outputs a deep semantic relevance score:

```
[Query Q] + [Document D1]  →  Neural Network  →  Score S1
[Query Q] + [Document D2]  →  Neural Network  →  Score S2
[Query Q] + [Document D3]  →  Neural Network  →  Score S3

```

The query and document are processed together (not independently), enabling the model to compute deep semantic relevance. Documents are then ordered by their scores. If D2 achieves the highest relevance score, D2 is declared the best match, overriding the initial ranking that placed D1 first.

Other re-ranking techniques include LLM-based re-rankers.

**Trade-off and best practice:** re-ranking improves accuracy but is computationally expensive. Computing deep relevance scores for every retrieved document takes time. Best practice: apply re-ranking only to the top subset — for example, the top 25 of 100 retrieved documents. This concentrates computational effort on the most promising candidates and reduces total time while still yielding a well-ranked final result.

**Common mistakes**

- Re-ranking all retrieved documents regardless of set size — this wastes computation on low-quality candidates. Re-rank only the top subset (e.g., top 25 of 100).
- Skipping re-ranking entirely to save time — initial retrieval results are often noisy; re-ranking significantly improves the context quality the LLM receives.

---

### Scalable RAG system design

A scalable RAG system at big-data scale combines all the components into a single pipeline that addresses the core scale challenges.

**Why it matters**

Understanding how the pieces fit together lets you design, debug, and extend production-grade RAG systems. Each stage builds on the previous one, and a bottleneck at any stage can defeat the entire pipeline.

**Walkthrough**

```
Data ingestion and ETL(Apache Spark cluster)Distributed vector storage(Vector database)Efficient ANN search(LSH reduces search space)Hybrid retrieval(BM25 + vector search)Re-ranking(Cross-encoder or LLM)LLM generates final answer
```

The six stages of a scalable RAG system:

1. 
**Data ingestion and ETL at scale** — Apache Spark distributes document loading, cleaning, chunking, and embedding computation across a cluster. One million PDFs, for example, are split across worker nodes; each node computes embeddings in parallel.

2. 
**Distributed vector storage** — embeddings are stored in a vector database with efficient indexing.

3. 
**Efficient search** — ANN techniques (such as LSH) replace linear search, reducing search space from billions of comparisons to a single bucket.

4. 
**Hybrid retrieval** — combining BM25 keyword search and vector/semantic search improves recall and precision.

5. 
**Re-ranking** — a cross-encoder or LLM re-ranker refines the initial retrieval results before they are passed to the LLM.

6. 
**LLM generation** — the LLM generates a final answer using the retrieved and re-ranked context.

This design addresses the scale challenges of embedding generation cost, query latency, storage, and multi-user traffic by applying distributed computing at each stage.

**Common mistakes**

- Designing each stage without considering how outputs flow into the next — a vector format mismatch between the embedding step and the vector database will cause silent failures at query time.
- Over-engineering the pipeline for a small dataset — apply Spark and ANN only when the scale genuinely requires it; premature optimization adds complexity without benefit.

---

## 3. Key Takeaways

- RAG solves three core LLM limitations — hallucination, no access to private data, and expensive fine-tuning — by retrieving up-to-date context at query time without retraining.
- Apache Spark enables distributed embedding generation at scale: documents are partitioned across executor nodes and each node computes embeddings in parallel. Transformations are lazy; an action like `.collect()` triggers execution.
- LSH converts nearest-neighbor search into a bucket lookup: each vector is hashed into a binary bucket ID, and a query only searches one of 2^n buckets. Approximation error occurs only at bucket boundaries.
- Hybrid search (BM25 keyword search + ANN vector search) captures both exact-term matches and semantically related content, producing better retrieval results than either method alone.
- Re-ranking adds a quality gate after initial retrieval by scoring query-document pairs through a cross-encoder; apply it to the top subset (e.g., top 25 of 100) to balance accuracy and computational cost.

**Mental model:** Think of a scalable RAG system as a multi-stage funnel. Spark fills it with embeddings; LSH narrows the search space to a single bucket; hybrid search casts a wide but targeted net; and re-ranking polishes the final results before they reach the LLM.

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