# 75. Distributed Embeddings - Suman - 8 May 2026

# RAG at Scale — Distributed Embeddings, Inference Optimisation, and Production Architecture

## 1. What You'll Learn in This Section

In this lesson, you'll learn to:

- Explain why a basic RAG system needs a distributed pipeline to handle millions of documents at low latency.
- Build a distributed embedding pipeline using PySpark, from creating a SparkSession to applying UDFs across partitions.
- Compare keyword search, semantic search, hybrid search, and re-ranking — and describe when each technique is used in a production pipeline.
- Identify the key inference optimisation techniques (KV caching, speculative decoding, tensor parallelism, batch parallelism, chunk prefill) and explain what each one improves.

## 2. Detailed Explanation

### Why scaling RAG is necessary

A basic RAG prototype works well when a small number of documents are uploaded and queried. Production RAG is fundamentally different.

**Why it matters**

Real organisations already hold large corpora of documents — policies, product data, account records — that live in a database. Users authenticate and query directly; they do not upload documents. For example, a bank chatbot already holds all FD (fixed deposit) rate documents for every customer. The user logs in and asks questions; no manual upload is needed. These document corpora update continuously — daily, hourly, or weekly — so the ingestion pipeline must run on a recurring schedule (for example, via a cron job) to keep the vector store current. At scale the system may deal with millions of documents. A system that takes 10 minutes to answer a query is unusable. Low latency at document-scale is the core motivation for bringing Apache Spark, GPU acceleration, and advanced inference techniques into the RAG architecture.

**Walkthrough**

The fundamental problem is scale:

- Sequential processing of a large new document batch without distributed computing takes 8–12 hours.
- Production RAG must answer queries at low latency even while the knowledge base contains millions of documents.
- The ingestion pipeline must run on a recurring schedule so the vector store stays current.

**Common mistakes**

- Assuming a prototype RAG system scales automatically to production — it does not. New infrastructure, distributed processing, and inference optimisations are all required.
- Forgetting that the ingestion pipeline must run on a schedule. A one-time embedding run leaves the knowledge base stale as documents change.

---

### Production RAG architecture

A production RAG system has two parallel tracks that converge at query time: a data ingestion pipeline and a query pipeline.

**Why it matters**

Understanding the two-track architecture helps you see where each technique (Spark, hybrid search, re-ranking, inference engines) fits. Every optimisation targets a specific stage.

**Walkthrough**

The **data ingestion pipeline** runs independently of user queries:

1. Documents arrive from object storage (such as Amazon S3) or databases — PDFs, CSVs, Excel files.
2. A data-processing pipeline cleans, chunks, and embeds the documents using distributed (Spark-driven) embedding.
3. The resulting embeddings are written to a vector database, which becomes the up-to-date knowledge store.
4. This pipeline runs on a schedule; each cycle replaces or adds to existing vectors.

The **query pipeline** runs at query time:

1. The user submits a query.
2. Hybrid search retrieves the top-k most relevant chunks from the vector database.
3. A re-ranker re-orders the top-k results using a cross-encoder model.
4. The top re-ranked chunks (for example, top 3) are passed as context to an LLM generator.
5. The LLM generates the final answer and returns it to the user.

The system sits on containerised services (Docker), orchestrated by Kubernetes, with GPU nodes for model hosting, auto-scaling policies, caching, and object storage.

```
Raw Documents(S3, DBs)Ingestion Pipeline(Spark, chunking, embedding)Vector Database(knowledge store)User QueryHybrid Search(keyword + semantic)Re-Ranker(cross-encoder)LLM GeneratorFinal Answer
```

**Common mistakes**

- Confusing the ingestion pipeline with the query pipeline — they run on different schedules and serve different purposes.
- Thinking the LLM queries the document store directly. The vector database is queried first; the LLM only receives the top re-ranked chunks as context.

---

### The chunking process

**Chunking** is the step that breaks a large document into smaller pieces before embedding.

**Why it matters**

Embedding models have a maximum input length. A long document cannot be embedded as a single vector. Chunking ensures every piece of text is small enough to embed and retrieve precisely.

**Walkthrough**

In a demo using 200 synthetic research articles (~246 words and ~1,700 characters each):

- **Chunk size:** 80 words
- **Overlap:** 15 words — each new chunk starts 15 words before the end of the previous chunk

The overlap preserves context at the boundary between adjacent chunks. Without it, a sentence split across two chunks might lose meaning at the join. Each pair of consecutive chunks shares 15 words, so no sentence is cut off and orphaned.

Each chunk is stored in the vector store as an embedding (a numeric vector, for example `[0.8, 0.2, 0.5, 0.9, ...]`) with metadata fields (author, upload date, page number, source) stored alongside it.

**Common mistakes**

- Setting overlap to zero. Adjacent chunks then share no context, so a sentence split at a boundary is broken across two chunks with no recovery.
- Using a chunk size that is too large. Overly long chunks dilute the relevance signal when retrieved.

---

### Apache Spark and distributed processing

**Apache Spark** is a general-purpose distributed processing framework that processes data in parallel by partitioning the dataset across multiple worker nodes (executors) that run simultaneously.

**Why it matters**

Traditional sequential processing (for example, pandas-based pipelines) processes documents one at a time. At millions of documents, this becomes prohibitively slow. When a manager delivers 20,000 new documents on a Friday evening and requests immediate ingestion, sequential embedding would take 8–12+ hours. Distributed embedding with Spark processes all documents in parallel across worker nodes and completes the same task in a fraction of the time — supporting near real-time indexing.

Spark is not RAG-specific. It is used across big data, ML modelling, NLP, and computer vision pipelines. It is included here because production RAG deals with document volumes that require parallel processing.

**Walkthrough**

Spark advantages relevant to the RAG ingestion pipeline:

- Integrates natively with cloud object stores: Amazon S3, Azure Data Lake Storage (ADLS), and Google Cloud Storage (GCS).
- Supports cloud-native AI pipelines (Azure ML, AWS SageMaker, Bedrock).
- Provides partition tuning and caching to improve job performance.
- Widely adopted in finance, healthcare, and telecom AI systems.

**Pandas vs Spark comparison:**

Approach | Processing model | Suitable scale
pandas | Sequential, single node | Small datasets
PySpark | Distributed, multiple nodes in parallel | Millions of documents (crores)

For a corpus of crores (tens of millions) of documents, sequential processing could take 8–12 hours. Distributed processing with Spark compresses this significantly.

**Common mistakes**

- Running Spark locally on a single machine and expecting linear speedup. A single-node local Spark job does not exploit multi-node parallelism — it actually adds overhead compared to pandas.
- Using pandas for document ingestion in production RAG without realising it will time out or run out of memory at scale.

---

### PySpark and SparkSession

**PySpark** is the Python API for Apache Spark — a library that exposes Spark capabilities from Python code, in the same way SQLAlchemy exposes SQL or other wrapper libraries expose non-Python frameworks.

**Why it matters**

You write Python to control a distributed cluster. Without PySpark, you would need to write in Scala or Java to use Spark. PySpark makes distributed processing accessible from the same environment where data scientists already work.

**Walkthrough**

To use PySpark:

1. Install PySpark inside the virtual environment.
2. Set the PySpark path in OS environment variables so Python can locate the Spark installation.
3. Create a `SparkSession` with appropriate configuration before running any Spark operations.

PySpark syntax differs from pandas. It resembles SQL-style operations and imports its types and functions from `pyspark.sql` modules.

Creating a `SparkSession`:

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("RAGScalingDemo") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.sql.adaptive.enabled", "true") \
    .getOrCreate()

```

Key configuration parameters:

- `spark.driver.memory`: memory allocated to the driver (for example, 4 GB).
- `spark.sql.shuffle.partitions`: number of partitions for shuffle operations (for example, 8).
- `spark.sql.adaptive.enabled`: enables adaptive query execution.

**Common mistakes**

- Forgetting to set environment variables before importing PySpark — this causes a runtime error when Python cannot locate the Spark installation.
- Creating the `SparkSession` more than once in the same script. Use `.getOrCreate()` so it reuses an existing session rather than creating a new one.

---

### Partitions, schemas, and UDFs in PySpark

**Partitions** are the units into which Spark divides your data. Each partition is assigned to a worker (executor), and all executors process their partition simultaneously.

**Why it matters**

Partitions are the mechanism that makes Spark parallel. The more partitions you have (up to the number of available executors), the more work runs simultaneously. A benchmark using 200 documents configured 12 partitions — meaning those documents were read and processed across 12 parallel workers.

**Walkthrough**

Unlike pandas (where `pd.DataFrame(data)` infers the schema), PySpark requires an explicit schema definition before creating a DataFrame:

```python
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

schema = StructType([
    StructField("doc_id", StringType(), True),
    StructField("chunk_id", IntegerType(), True),
    StructField("chunk_text", StringType(), True)
])

```

The schema declares column names and data types. The DataFrame is then created by passing the chunked data alongside this schema.

**UDFs (User-Defined Functions)** allow custom Python functions — such as a chunking or embedding function — to be applied across every partition in parallel. This is how PySpark enables distributed embedding generation: the embedding function is registered as a UDF and applied to the distributed DataFrame, so each executor runs the function on its partition independently.

**Common mistakes**

- Forgetting the schema definition and trying to create a Spark DataFrame the same way as a pandas DataFrame — PySpark requires the schema to be declared explicitly.
- Writing a UDF that imports a large model inside the function body without checking whether the model is already loaded.
- This causes the model to be re-loaded on every row instead of once per partition.

---

### Embeddings and distributed embedding

**Embeddings** convert text into vector representations that preserve the semantic (meaning-based) relationships of the text.

**Why it matters**

Embeddings let the system find documents that are conceptually related to a query — not just documents that share the same keywords. For example, the words "king" and "man" share no common letters but are semantically related through latent properties (dimensions). Embeddings capture this latent structure so that the relationship between concepts becomes mathematically computable.

**Walkthrough**

The embedding model used in this benchmark is `sentence-transformers/all-MiniLM-L6-v2` — a small, CPU-compatible sentence transformer model. Weights are loaded into memory when the model is instantiated via `SentenceTransformer`.

Sequential embedding results (without Spark):

Chunks processed | Time
200 chunks | ~7 seconds
400 chunks | ~14 seconds
800 chunks | ~31–32 seconds

PySpark results (12 partitions):

Stage | Time
Reading 200 documents | ~4.4 seconds
Chunking (PySpark pipeline) | ~16 seconds
Embedding | ~26 seconds

The demo notebook ran locally rather than on a true multi-node cluster. At scale on a real cluster, PySpark is significantly faster than sequential processing — especially at higher document counts.

**Common mistakes**

- Confusing the sequential notebook results (which showed Spark appearing slower due to local execution overhead) with the intended behaviour at scale.
- On a real multi-node cluster, distributed embedding is far faster than sequential processing.
- Using a large, GPU-only embedding model on CPU worker nodes. Choose a CPU-compatible model (such as `all-MiniLM-L6-v2`) when running on CPU executors.

---

### Embedding aggregation and multilingual embeddings

After all worker nodes generate embeddings in parallel, the results must be combined into a single coherent dataset.

**Why it matters**

Distributed processing produces results spread across multiple nodes. Without a structured aggregation step, the embeddings are scattered and cannot be written consistently to the vector store.

**Walkthrough**

Aggregation uses a **shuffle-and-merge** technique, followed by an **ordered embedding collection** strategy to align all embeddings consistently before writing them to the vector store.

**Multilingual and domain-specific embeddings** are a distinct use case. Standard commercial models (Gemini, ChatGPT, and similar) are predominantly trained on international (non-Indic) languages. Building chatbots in Telugu, Tamil, Hindi, Punjabi, and other Indic languages requires:

- Fine-tuning local Indic language models rather than using standard cloud LLMs directly.
- Adopting a different schema for embedding and retrieval.
- Testing with native-language question sets.

This requires an adjusted pipeline design compared to standard English-language RAG.

**Common mistakes**

- Skipping the ordered embedding collection step. Without it, embeddings from different workers may be written in an inconsistent order, causing mismatches between chunk IDs and their stored vectors.
- Applying a standard English embedding model to Indic-language documents and expecting meaningful retrieval. Semantic similarity in another language requires a model trained on that language.

---

### Keyword search and semantic search

Two fundamental retrieval approaches exist before hybrid search combines them.

**Why it matters**

Understanding their individual limitations explains why hybrid search is the industry standard.

**Walkthrough**

**Keyword search** matches exact words (and their variations) from the user query against documents in the store. It underlies traditional search engines.

Limitation: if the user's query uses different vocabulary than the stored documents, keyword search misses relevant results. It also does not understand negation or semantic intent — a query like "climate change but not renewable energy reduction" would still surface documents containing all those keywords.

**Semantic (vector) search** uses embeddings to find documents that are conceptually similar to the query, even without shared exact keywords. For example, a document about "combating global warming" is semantically related to a query about "climate change reduction" even without lexical overlap.

**Common mistakes**

- Using only keyword search in a production RAG system. Queries phrased differently from stored documents will return poor results.
- Using only semantic search and ignoring exact-match needs. Some queries require precise keyword matching (product codes, dates, names) that semantic search handles less reliably.

---

### Hybrid search

**Hybrid search** combines keyword search and semantic search by merging their result lists before passing candidates to the re-ranker.

**Why it matters**

Neither pure keyword search nor pure semantic search performs well across all query types. Hybrid search is the industry standard for production RAG because it captures both exact-match relevance and conceptual similarity.

**Walkthrough**

How hybrid search works in practice:

1. Keyword search returns a ranked list of document IDs.
2. Semantic (vector) search returns a separate ranked list of document IDs.
3. The two lists are concatenated — for example, top-5 from keyword + top-5 from semantic = top-10 candidates.
4. The top-k candidates are passed to the re-ranker.

```
User QueryKeyword Search(top-5 IDs)Semantic Search(top-5 IDs)Merge(top-10 candidates)Re-Ranker(cross-encoder)Top-3 Chunksto LLM
```

**Common mistakes**

- Concatenating the two lists without deduplication. A document ID appearing in both lists should be present only once in the merged candidate set.
- Using a very large top-k for hybrid search before re-ranking. More candidates mean higher re-ranking compute cost — balance recall with efficiency.

---

### Metadata filtering

**Metadata filtering** is a pre-retrieval step that narrows the candidate set before hybrid search runs.

**Why it matters**

On a corpus of 1 crore (10 million) documents, running hybrid search on every document is extremely expensive. Filtering down to 20,000 candidates (for example, year = 2020 AND country = India) makes the subsequent search far cheaper and faster.

**Walkthrough**

How metadata filtering works:

1. During embedding ingestion, metadata fields are stored alongside embeddings in the vector store. Example fields: document ID, title, year, country, source.
2. At query time, entities are extracted from the user's query (year = 2020, country = India, source type = government) using regex, Python logic, or an LLM call.
3. A filter is applied to the vector store: return only records matching the extracted metadata.
4. Hybrid search then runs only on the filtered subset.

Metadata for extraction can come from:

- The PDF filename.
- The first page of a PDF (which typically contains author, publication date, title, preface).
- LLM extraction from the first page.

Metadata filtering is a retrieval-stage technique, not an inference-engine optimisation.

**Common mistakes**

- Storing metadata fields inconsistently across documents. If year is stored as a string in some documents and as an integer in others, the metadata filter will silently miss records.
- Extracting metadata only at query time rather than at ingestion time. Metadata should be extracted and stored during ingestion so it is immediately available for filtering.

---

### Re-ranking with cross-encoders

**Re-ranking** re-evaluates and re-orders the hybrid search candidates using a separate, more accurate model before passing results to the LLM.

**Why it matters**

The retriever's ranking heuristic is approximate. A document ranked 1st by hybrid search may not actually be the most relevant — a re-ranker assesses each (query, document) pair independently and corrects the order. This improves the quality of the context passed to the LLM.

**Walkthrough**

The re-ranker is typically a cross-encoder neural network or an LLM. Example walkthrough:

1. Hybrid retriever returns top-50 candidates, ranking document 4 as most relevant and document 10 as least relevant.
2. Re-ranker assesses all 50 independently and determines that document 7 is actually most relevant.
3. The pipeline selects the top 3 from the re-ranked list.
4. The top 3 chunks (paragraph-length or table-length) are bundled with the user's original query and passed to the LLM, which generates the final answer from the provided context.

**Common mistakes**

- Running the re-ranker on the entire corpus instead of the top-k candidates from hybrid search.
- Re-ranking every document is far too slow — always use it as a second-stage filter on a small candidate set.
- Passing too many re-ranked chunks to the LLM. More context is not always better — it increases token cost and can confuse the model.
- The top 3 (or a similarly small number) is the standard upper limit to pass as context.

---

### The inference engine concept

An **inference engine** is a runtime that wraps an LLM and makes it callable, efficient, and production-safe.

**Why it matters**

An LLM cannot be called directly as a raw model file. Think of an LLM as the engine of a car: you cannot drive a car by inserting a key directly into the engine. The engine must be wrapped in the body, controls, transmission, and other systems that make it driveable. The inference engine is that wrapper. Without it, the LLM is not callable or controllable.

**Walkthrough**

When a user query arrives at the inference engine:

1. The engine forwards the query to the underlying LLM.
2. The LLM generates a response.
3. The engine returns the response to the user.

The inference engine's configurations determine latency, cost, throughput, and how well the LLM handles concurrent requests. Failing to implement optimisations does not prevent a RAG system from working, but it causes high latency, inability to serve multiple concurrent users, and excessive infrastructure cost.

**Common mistakes**

- Calling the LLM directly (for example, via a raw HuggingFace `model.generate()` call) in production without an inference engine. This approach provides no batching, caching, or concurrency management.
- Treating inference engine configuration as a one-time setup. Tuning parameters (batch size, cache size, parallelism settings) need to be revisited as traffic and model size change.

---

### KV caching

**KV (key-value) caching** stores the key-value states computed for previously processed tokens so they can be reused in subsequent turns of the same conversation.

**Why it matters**

In a multi-turn conversation, earlier turns contain context that should inform later responses. Without KV caching, the LLM recomputes attention over all previous tokens from scratch at every new query. This is slow and wasteful.

Think of a friend asking "was it spicy?" mid-conversation about today's lunch. You do not ask for clarification — you already hold the context in working memory. KV caching gives the LLM the same ability to retain context across turns.

**Walkthrough**

- Without KV caching: at turn 3 of a conversation, the LLM re-processes turns 1 and 2 from scratch before generating a response.
- With KV caching: the LLM stores intermediate computations from turns 1 and 2 and picks up from where it left off at turn 3.

Benefits: reduced latency on follow-up queries, lower compute cost, better context retention.

**Common mistakes**

- Allocating too little memory for the KV cache. If the cache fills up, older context is evicted, and the model loses earlier conversation turns.
- Confusing KV caching (inference-engine level) with infrastructure-level caching (storing full query results). They operate at different layers and serve different purposes.

---

### Speculative decoding

**Speculative decoding** uses two models — a small draft model and a large verification model — to generate tokens faster.

**Why it matters**

Standard token generation is sequential: the large model produces one token at a time. Speculative decoding parallelises drafting and verification to increase generation speed.

Think of it like a developer and an intern: the intern (draft model) writes code first; the senior developer (large model) reviews and accepts the correct portions. The developer's time is spent validating rather than generating from scratch.

**Walkthrough**

1. A small, fast **draft model** speculatively generates multiple tokens ahead.
2. A large **verification model** checks the draft tokens and accepts the correct ones.
3. The large model verifies in parallel with the draft model's generation.

Result: more tokens are generated per step, increasing overall generation speed.

**Common mistakes**

- Choosing a draft model that is too large. The draft model should be significantly smaller and faster than the verification model; otherwise the speedup disappears.
- Using speculative decoding when the draft model produces many incorrect tokens. If the large model rejects most drafts, the overhead of verification exceeds the savings from speculative generation.

---

### Tensor parallelism, batch parallelism, and chunk prefill

These three techniques address different bottlenecks in LLM inference.

**Why it matters**

Large LLMs often cannot fit on a single GPU. Concurrent user traffic stresses the model's throughput. Long context documents slow down the time to first token. Each technique targets one of these problems.

**Walkthrough**

**Tensor parallelism** splits the model's weight matrices across multiple GPUs. Different GPUs handle different portions of the computation in parallel, then combine their outputs. Input tokens are split across GPU nodes; each node performs its portion; results are concatenated to produce the next-token prediction.

**Batch parallelism** processes multiple user requests together in a single forward pass. Instead of serving one user query at a time, the engine batches N concurrent queries and runs them through the model simultaneously. This improves GPU utilisation and overall throughput.

**Chunk prefill** breaks a long context document into chunks and begins feeding them into the model one at a time rather than loading the entire context before starting generation. Token generation begins as soon as the first chunk is filled, instead of waiting for the entire context to load. This reduces **TTFT (time to first token)** and lowers memory pressure.

Technique | Primary benefit
Tensor parallelism | Splits model weights across GPUs; parallelises computation
Batch parallelism | Serves multiple users simultaneously; improves GPU utilisation
Chunk prefill | Reduces TTFT; lowers memory pressure for long-context documents

**Common mistakes**

- Applying tensor parallelism without coordinating the split size with the number of available GPUs. Mismatched configuration can lead to out-of-memory errors on some nodes.
- Skipping chunk prefill for long-context RAG documents. Without it, the model waits for the entire document context to load before producing a single token.

---

### GPU scheduling and cluster tuning

**GPU scheduling** is the policy that determines how GPU resources in a cluster are allocated across workloads and users.

**Why it matters**

RAG involves multiple heavyweight models: a guardrail model, a pre-trained tokeniser, an embedding model, and an LLM generator. Loading these models onto CPU and running inference is extremely slow. GPU nodes load models into fast GPU memory and perform inference much faster. Without scheduling, some team members exhaust all available GPUs while others cannot run experiments at all.

**Walkthrough**

A GPU cluster contains multiple GPU nodes distributed across availability zones (for example, AWS regions: US East 1, US East 2, India). Scheduling policies include:

- **Fair sharing**: every user or tenant gets a proportional share of GPU resources.
- **Gang scheduling**: schedules a group of tasks that must run together simultaneously.
- **Preemption**: can interrupt lower-priority jobs to free resources for higher-priority ones.
- **Resource-aware scheduling**: places workloads on nodes that have the right resources.
- **Priority queue**: orders workloads by priority when resources are contended.

Goals of a GPU scheduling policy: maximum GPU utilisation, meeting SLOs (service-level objectives), fairness across all tenants, and cost efficiency.

GPU scheduling and cluster tuning is the responsibility of an infrastructure (infra) engineer. An AI engineer should understand what GPU scheduling is and why it matters, but is not expected to implement scheduling policies independently.

**Common mistakes**

- Assuming every workload gets equal GPU resources by default. Without a fair-sharing policy configured, unmanaged workloads will consume resources in an unpredictable order.
- Conflating GPU scheduling (cluster-level resource allocation) with tensor parallelism (splitting a single model across GPUs). They operate at different layers.

---

### Cloud production architecture on AWS

A production RAG deployment uses a set of coordinated AWS services.

**Why it matters**

Understanding which service fills which role lets you design or describe a production system clearly — for example, in system design interviews or architecture discussions.

**Walkthrough**

Key AWS services in a production RAG deployment:

Service | Role
Amazon S3 | Object storage for raw documents (PDFs, CSVs, Excel)
Amazon ECR | Registry for Docker images (analogous to Docker Hub)
Amazon ECS / ECS Fargate | Runs containerised applications; Fargate is the serverless cluster variant
EC2 | Virtual machines for running application workloads
EKS | Managed Kubernetes cluster for orchestration
Application Load Balancer (ALB) | Distributes traffic; supports auto-scaling
VPC | Isolated network with public and private subnets across availability zones
CloudFront | CDN (content delivery network) service
ElastiCache | Managed caching layer (for example, Redis)
SQS | Message queue for decoupling pipeline stages
AWS CloudWatch | Logs, metrics, and monitoring dashboards
Helm charts | YAML-based one-click deployment mechanism for Kubernetes workloads

**CI/CD pipeline flow:**

1. Developer writes application code (FastAPI backend, React frontend, RAG service, worker service).
2. Code is pushed to GitHub.
3. CI/CD pipeline (for example, GitLab CI) builds and tests the code in a dev/UAT environment.
4. Docker images are built from the tested code.
5. Images are pushed to Amazon ECR.
6. Deployment to ECS or EKS cluster using Helm charts.
7. Runtime logs and metrics are monitored via CloudWatch.

**Common mistakes**

- Pushing Docker images directly to a running ECS cluster without going through ECR. ECR is the correct registry; bypassing it breaks the CI/CD contract and audit trail.
- Forgetting that SQS decouples pipeline stages. Without a message queue, a spike in document uploads can overwhelm the embedding workers directly.

---

### Auto-scaling, observability, and infrastructure-level caching

**Auto-scaling** adjusts the number of running service instances based on current user load.

**Why it matters**

At peak times, more machines are provisioned to handle the load. During off-peak hours, the system scales down, reducing infrastructure cost. Without observability, you cannot tell when the system is struggling or where the bottleneck is.

**Walkthrough**

**Observability metrics** tracked in production RAG:

Category | Example metrics
Performance | Latency, throughput, TTFT (time to first token), ITL (inter-token latency)
Cost | Cost per query

Tools used: Prometheus, Grafana, Alert Manager, AWS CloudWatch.

**Infrastructure-level caching** (distinct from KV caching) stores the results of previous pipeline operations so they can be served immediately on repeat queries without re-running the full pipeline. Spark also supports caching of intermediate DataFrames within a job to prevent redundant recomputation across pipeline stages.

**Common mistakes**

- Tracking only latency and ignoring cost per query. At scale, a system that is fast but expensive per query may be unsustainable.
- Confusing infrastructure-level caching (storing full query results) with KV caching (storing intermediate LLM attention states). They are separate mechanisms at different layers of the stack.

---

## 3. Key Takeaways

- Sequential processing cannot scale to millions of documents. PySpark distributes chunking and embedding across multiple worker nodes in parallel.
- Distributed processing compresses 8–12 hours of sequential ingestion work into a fraction of the time.
- Hybrid search (keyword + semantic) is the production standard because neither approach alone handles all query types well.
- A re-ranker re-orders the top-k hybrid candidates using a cross-encoder model before passing the top results as context to the LLM.
- Metadata filtering reduces a corpus of millions to a few thousand candidates before hybrid search runs — dramatically cutting computation cost at scale.
- An LLM needs an inference engine wrapper to be callable in production.
- Five techniques target different bottlenecks: KV caching (context retention), speculative decoding (generation speed), tensor parallelism (compute distribution), batch parallelism (concurrent throughput), and chunk prefill (time to first token).
- Production RAG uses a layered infrastructure: Spark for ingestion, a vector database as the knowledge store, GPU nodes for model hosting, and Kubernetes for orchestration.
- AWS services (S3, ECR, ECS/EKS, CloudWatch) cover storage, container registry, compute, and monitoring at each layer.

**Mental model:** Think of an LLM as the engine of a car — powerful but unusable without the body, controls, and transmission wrapped around it. Production RAG is that full car: the ingestion pipeline feeds the fuel (documents), the query pipeline steers to the answer, and the inference engine makes the whole system driveable.

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