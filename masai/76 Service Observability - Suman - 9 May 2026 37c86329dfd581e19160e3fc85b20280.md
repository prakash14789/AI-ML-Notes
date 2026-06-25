# 76. Service Observability - Suman - 9 May 2026

# Service Observability — RAG at Scale, Deployment, and Production Monitoring

## 1. What You'll Learn in This Section

In this lesson, you'll learn to:

- Explain why PySpark outperforms sequential Python at large scale and when the trade-off flips.
- Build a distributed ingestion pipeline that chunks, embeds, and indexes documents using PySpark.
- Deploy a RAG application on AWS using Docker, ECR, Fargate, and an Application Load Balancer.
- Monitor a live RAG system with CloudWatch, tracking both infrastructure health and answer-quality metrics.

---

## 2. Detailed Explanation

### PySpark vs Sequential Ingestion — the Startup Trade-off

**PySpark** is the Python API for Apache Spark, a distributed processing framework. Before it processes any data, it must start a Spark session — loading libraries, establishing executors, and setting up partitions. This startup overhead is the key trade-off against plain sequential Python.

**Why it matters**

A production RAG system may need to index millions of PDFs. Sequential Python processes them one at a time. PySpark splits the work across many machines running in parallel. The question is: when does PySpark's parallelism outweigh its startup cost?

**Walkthrough**

Think of starting PySpark like starting a vehicle. The vehicle does not immediately reach top speed — it accelerates from 0 to 10, to 30, to 100 gradually. Sequential Python is like walking: it starts moving instantly but never goes faster than one step at a time. For a very short trip (small dataset), the time spent accelerating erases any speed advantage. For a long trip (millions of documents), the vehicle's top speed makes the total journey far shorter.

An experiment with 200 fake PDF documents (across categories: biotech, energy, AI, research) illustrated this clearly:

- Sequential total time: approximately 3.86 seconds.
- PySpark distributed chunking time: approximately 3.12 seconds for chunking alone, but the total pipeline time was higher because session startup was not yet amortised.

For 200 documents, sequential was faster overall. At 1,000 documents, PySpark catches up to sequential at roughly the 80-second mark, then pulls ahead. At 1 million or 10 million documents, PySpark processes what sequential could only handle in days, in minutes.

Two fairer metrics were introduced to avoid biasing the comparison:

Metric | Definition
Initialisation latency | How quickly the first document is processed once the session is running
Processing throughput | Documents (or chunks) processed per second, omitting startup time

Simple total-time comparison unfairly penalises PySpark because startup overhead inflates its time for small datasets.

**Common mistakes**

- Using total elapsed time as the only benchmark when comparing PySpark and sequential: this misrepresents PySpark's per-document speed once the session is warm.
- Choosing PySpark for small datasets where the startup cost is never amortised: use sequential Python when the document count is in the hundreds.

---

### PySpark Ingestion Code

PySpark ingestion has three steps: configure the Spark session, chunk documents using a UDF, then embed chunks using a Pandas UDF.

**Why it matters**

Writing this code correctly means the same logic that runs on a laptop notebook can scale to a cluster with hundreds of workers, without restructuring the code.

**Walkthrough**

**Step 1 — Configure the Spark session**

```python
# Building a Spark session with resource settings
spark = (SparkSession.builder
         .appName("RAG scaling demo")
         .config("spark.master", ...)
         .config("spark.executor.memory", ...)
         .config("spark.sql.shuffle.partitions", ...)
         .getOrCreate())

```

The session configuration specifies: application name (`RAG scaling demo`), master settings, memory allocation per executor, number of shuffle partitions, and whether adaptive query execution is enabled.

**Step 2 — Chunk with a UDF and `explode`**

```python
# Define a UDF that returns an array of chunk strings
chunk_udf = udf(chunk_text, ArrayType(StringType()))

# Apply the UDF and explode arrays into individual rows
chunked_df = (raw_df
              .select("doc_id", explode(chunk_udf(col("content"))).alias("chunk")))

```

`explode` converts an array column (one row holds all chunks for one document) into multiple individual rows — one row per chunk. Without `explode`, the chunks sit in a nested array that cannot be operated on row-by-row.

**Step 3 — Embed with a Pandas UDF**

```python
# Apply embedding function per partition using a Pandas UDF
embedded_df = (chunked_df
               .repartition(num_cores)
               .withColumn("embedding", embed_pandas_udf(col("chunk"))))

embedded_df.select("doc_id", "embedding").show(3, truncate=50)

```

PySpark display syntax uses `.show(n, truncate=N)` rather than Python's `df[["col"]].head(3)`.

The embedding model used is `sentence-transformers/all-MiniLM-L6-v2`, a lightweight Hugging Face sentence transformer that produces **384-dimensional vectors** — every chunk becomes a list of 384 floating-point numbers.

**Common mistakes**

- Mixing embedding models between document ingestion and query time: both must use the same model (`all-MiniLM-L6-v2` in this demo). Mixing models produces incoherent similarity scores.
- Forgetting `explode` after the UDF: the result is a nested array column that downstream operations cannot iterate over row-by-row.

---

### GPU-Aware Scheduling in PySpark

In a production Spark cluster, multiple worker nodes each have multiple GPUs. Efficient GPU use requires discovery, allocation, and task granularity configuration.

**Why it matters**

If a single task monopolises a GPU's VRAM, other tasks stall. If GPUs sit idle, compute cost is wasted. The `spark.task.resource.gpu.amount` setting controls this balance.

**Walkthrough**

```python
# GPU scheduling configuration (production pattern)
gpu_config = {
    "spark.task.resource.gpu.amount": "0.25"   # 4 tasks share one GPU
}

```

Setting `spark.task.resource.gpu.amount` to `0.25` means four tasks share one GPU. This prevents any single task from monopolising VRAM while ensuring no GPU sits underutilised.

The three steps Spark follows are:

1. **Discovery** — Spark identifies how many GPUs exist on each worker node.
2. **Allocation** — Spark assigns a specific number of GPUs to each executor.
3. **Task granularity** — Spark decides how many GPUs each task within an executor receives.

This configuration is handled by infrastructure engineers or data engineers in practice. Data scientists and AI engineers should be able to describe it in an interview, even if they do not set it themselves. Running on a single-GPU environment (such as a Colab notebook) means multi-GPU discovery is not exercised.

**Common mistakes**

- Over-subscribed VRAM (setting too many tasks per GPU): tasks fail with out-of-memory errors.
- Under-utilised GPUs (too few tasks per GPU): compute cost increases without a speed benefit.

---

### Chunking Strategy

**Chunking** is the process of splitting a document into smaller pieces that fit within an LLM's context window and capture a coherent unit of meaning.

**Why it matters**

A PDF may be hundreds of pages long. Embedding the whole document as one vector loses fine-grained meaning. Retrieving the right 80-word passage is far more precise than retrieving a 50-page document.

**Walkthrough**

Two key parameters control chunking:

- **Chunk size**: 80 words per chunk (as used in the notebook). A common production baseline is 500 tokens.
- **Chunk overlap**: 15–20 words (notebook) or 50–80 tokens (production baseline). Overlap ensures consecutive chunks share some words, preserving contextual continuity across boundaries.

The right overlap depends on experimentation. Overlap must not be so large that consecutive chunks become near-duplicates — for example, a 500-token chunk with 250-token overlap yields mostly duplicate content.

**`RecursiveCharacterTextSplitter`** from LangChain is the standard tool for applying chunk size and overlap in practice.

**Common mistakes**

- Setting overlap so high that chunks are near-duplicates: retrieval returns essentially the same content multiple times.
- Citing a single "correct" overlap value in an interview: the right answer is to cite the value found to work best through experimentation in a specific project.

---

### Vector Stores — Production vs Demo Choices

A **vector store** is a database designed to store and search high-dimensional embeddings efficiently.

**Why it matters**

Choosing the wrong vector store in a production context signals tutorial-level experience to an interviewer. The choice affects scalability, metadata filtering capability, and operational overhead.

**Walkthrough**

Demo vector stores (FAISS and ChromaDB) are lightweight and in-memory. They work well in notebooks and tutorials but lack persistence at scale and robust metadata filtering.

```python
import faiss, numpy as np

dimension = embeddings.shape[1]   # 384 for all-MiniLM-L6-v2
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)             # raw numpy embeddings

```

In the notebook demo, FAISS stores 384-dimensional vectors with document ID and chunk text as metadata. In production, embeddings are never double-stored in a Parquet file first — they are written directly to the vector store after generation.

Production vector store options:

Store | Notes
Milvus | Open-source, scalable
PGVector | Vector extension for PostgreSQL
Neo4j | Graph database; suited for knowledge-graph-based retrieval
Elasticsearch / OpenSearch | Widely adopted in enterprise
Azure AI Search | Azure's managed vector store
Weaviate | Cloud-native vector database
Pinecone | Managed vector database service

**Common mistakes**

- Mentioning FAISS or ChromaDB when describing a production RAG system in an interview: this signals a tutorial-level project to an interviewer.
- Double-storing embeddings in a Parquet file before writing to the vector store: in production, write directly to the store.

---

### RAG Retrieval Flow

The **RAG retrieval flow** is the sequence of steps that converts a user query into a grounded LLM response using retrieved document chunks.

**Why it matters**

Getting this flow right ensures the LLM answers from evidence rather than from its parametric memory. Each step — embedding, search, reranking, generation — must be configured for the production scale.

**Walkthrough**

```
User submits queryEmbed query with same model as documentsSimilarity search on vector store (L2 / cosine)Retrieve top-K chunks (K=2 notebook; K=5 deployed; 10-50 production)Reranker re-orders chunks by relevanceForward top 3-5 reranked chunks to LLMLLM generates grounded response (Claude Haiku)Response returned to user
```

Step-by-step:

1. Encode the query text with the same embedding model used during ingestion (`all-MiniLM-L6-v2`).
2. Run a similarity search on the vector store. Distance (L2 or cosine) determines closeness; lower distance means more similar.
3. Retrieve the top-K chunks. K = 2 in the notebook experiment; K = 5 in the deployed demo (all five reranked, top result used for generation); in production, K is typically 10–50.
4. Pass the top-K chunks through a reranker that re-orders them by relevance. Forward the top 3–5 reranked chunks to the LLM.
5. The LLM (Claude Haiku in the deployed demo) generates a response grounded in the retrieved context.
6. Return the response to the user.

```python
# Query embedding and search
query_vec = query_model.encode([query_text])        # same model as doc embedding
distances, indices = index.search(query_vec, k=2)   # top-2 chunks

# Reconstruct text from chunk index
chunk_text = chunks_df[indices[0]]

```

**Common mistakes**

- Using a different embedding model for queries than for documents: similarity scores become nonsensical and retrieval quality collapses.
- Setting K too low in production (e.g., K = 2): the reranker has too few candidates to choose from. Use K = 10–50 before reranking.

---

### Data Ingestion Pipeline — Production Pattern

A **production ingestion pipeline** is a separate, scheduled process that keeps the vector store current as source documents are added, updated, or removed.

**Why it matters**

A bank's interest rate PDFs change regularly. Without a pipeline that re-indexes changed documents, the chatbot answers from stale content. The pipeline runs independently of the user-facing service.

**Walkthrough**

The pipeline runs six steps each time it executes:

1. **Fetch** — connect to the document source (S3, NFS, cloud storage, or a database folder). Download all files with a timestamp newer than the last run.
2. **Deduplication / update check** — compare each downloaded file against what is already in the vector store. If a file exists with a newer timestamp, delete all old chunks for that document and re-process.
3. **Deletion of stale documents** — read a manifest of outdated documents, find matching entries in the vector store, and remove them.
4. **Chunking** — split each PDF using OCR (to extract text) and a chunking strategy (chunk size, overlap).
5. **Metadata extraction** — extract fields such as doc ID, author, timestamp, last-update date, current status, page number, paragraph number, and topic. LLMs or regex can be used for extraction.
6. **Embedding and upsert** — generate embeddings and write chunks + metadata to the vector store.

**Scheduling:** the pipeline runs as a cron job. Frequency depends on document sensitivity:

- Real-time / daily-updated domains (e.g., stock analysis): run once or twice a day.
- Slowly-changing domains (e.g., internal HR policies): run once per week or month.

In student projects, GitHub Actions can replicate a cron job by scheduling a workflow at a fixed time. On Linux systems, native cron jobs also work.

**Live upload path:** when a user uploads a PDF through the chatbot UI, the same ingestion logic runs in real time:

```
if user uploads a file:
    run ingestion pipeline steps (chunking → embedding → upsert)
else:
    retrieve directly from existing vector store

```

Both paths coexist. The background pipeline handles bulk updates; the live path handles individual user uploads.

**Common mistakes**

- Running ingestion every time a user submits a query: ingestion is a separate background process, not part of the query path.
- Skipping deduplication: re-indexing unchanged documents wastes compute and creates duplicate chunks in the vector store.

---

### Production Deployment on AWS

**Containerising a RAG application** and deploying it on AWS using managed services makes the service scalable, reproducible, and observable.

**Why it matters**

A local Streamlit app on a laptop cannot serve thousands of users. Deploying on Fargate with a load balancer means the service scales automatically and remains available even when individual containers fail.

**Walkthrough**

The deployment stack follows this sequence:

```
Developer IDE (VS Code)Write DockerfileBuild Docker image locally (Docker / Podman)Push image to ECR (Elastic Container Registry)CloudFormation stack: ECS cluster + Fargate + ALB + CloudWatchRun deploy shell script (deploy_eval_app.sh)Application URL served via ALBCloudWatch collects logs and metrics
```

**AWS services used:**

Service | Purpose
ECR | Private Docker image registry hosted on AWS
ECS | Container orchestration; ECR image registered as a task definition
Fargate | Serverless compute layer; runs ECS tasks without managing EC2 instances
ALB | Distributes incoming traffic; provides the public application URL
CloudWatch | Collects application logs, performance metrics, and custom telemetry
VPC | Network isolation layer; routes requests between services
CloudFormation | Infrastructure-as-code; groups ECS cluster, Fargate, ALB, and CloudWatch into one deployable stack

**Repository structure for production projects:**

- `src/` — main application logic (Python RAG code: retrieval, embedding, reranking, observability).
- `scripts/` — turnkey shell scripts for one-click deployment.
- `docker/` — Dockerfile(s) defining the container image.
- `README.md` — step-by-step replication guide.

The demo Docker image was 798 MB. When code changes, the image is rebuilt, pushed to ECR with an updated tag, and the deployment steps are repeated.

**Containerisation is cloud-agnostic.** The same Dockerfile runs locally or on any cloud provider. In Azure, ECR is replaced by Azure Container Registry (ACR); in Red Hat OpenShift, by Quay.

**Common mistakes**

- Rebuilding and redeploying the image without updating the tag: ECR may serve a cached image rather than the new one.
- Manually configuring each AWS service in the console instead of using CloudFormation: manual setup is hard to reproduce and error-prone.

---

### Cloud Market Share and Certification Guidance

Understanding which cloud platforms dominate the market helps an AI engineer focus certification effort where employer demand is highest.

**Why it matters**

Certifications on niche platforms yield fewer job opportunities. AWS and Azure together cover the majority of employer requirements for AI engineers and data scientists.

**Walkthrough**

Market share (approximate):

- **AWS**: more than 30% of the cloud market.
- **Azure**: approximately 22–23%.
- **GCP**: approximately 10%.
- IBM, Red Hat/OpenShift, NVIDIA, and Oracle hold smaller shares.

Equivalent services across the major clouds:

Concept | AWS | Azure | GCP | Other
Managed ML platform | SageMaker | Azure AI Foundry / Azure ML | Vertex AI | Watson X AI (IBM), OpenShift AI
Container registry | ECR | ACR | Artifact Registry | Quay (OpenShift)
Virtual machine | EC2 | Virtual Machine | Compute Engine | —
AI-native LLM platform | Bedrock | Azure AI Foundry | Vertex AI | —

**Recommended certifications:**

- AWS: AWS Certified Machine Learning Specialty; AWS AI Practitioner.
- Azure: AI-900, AZ-900, DP-100, AI-102.

AI tools (such as Cursor agent and Claude Code) now handle cloud-specific syntax, so memorising exact CLI commands is less critical. What matters is understanding configuration parameters: instance size, concurrency, scaling thresholds, and resource limits.

**Common mistakes**

- Memorising CLI syntax rather than understanding configuration concepts: tools generate syntax, but configuring resource limits and scaling thresholds requires conceptual understanding.

---

### Service Observability — CloudWatch Metrics

**Service observability** means having enough visibility into a running system to detect problems, understand their cause, and fix them quickly.

**Why it matters**

A RAG chatbot can fail in two distinct ways: the infrastructure can degrade (high CPU, slow responses) or the answer quality can degrade (hallucinations, irrelevant retrievals). Without separate metric categories for each, a team may miss answer-quality degradation entirely.

**Walkthrough**

CloudWatch tracks two categories of metrics:

**Operational (functional) metrics** — infrastructure health:

- CPU utilisation
- Memory utilisation
- Running task count
- Target response time (latency)
- Error rates

A hard threshold is set for each. When too many responses exceed the limit, an automated alert fires.

**Evaluation (answer-quality) metrics** — RAG answer correctness:

Metric | Meaning | Example threshold
Faithfulness | How well the generated answer is supported by the retrieved context. Scale: 0–1. | Do not let drop below 50%
Context precision | How precise the retrieved chunks are relative to the query | Do not let drop below 80%
Context coverage | How much relevant information in the knowledge base was covered | Do not let drop below 40%
End-to-end app latency | Total time from user query to final response | —
Search and rerank quality | Effectiveness of retrieval + reranking step | —
Sampling and failure rate | Frequency of failed or empty responses | —

A sample query on the deployed chatbot scored 96% faithfulness.

Each interaction logs: user query text, retrieved context (chunk content), LLM-generated answer, per-metric scores (faithfulness, context precision, context coverage), upload or artifact events, and telemetry from all running services. In production, logs are downloaded as JSON files and analysed programmatically.

**Common mistakes**

- Monitoring only infrastructure metrics and ignoring answer-quality metrics: a system can have green CPU and latency dashboards while silently producing hallucinated answers.
- Reading CloudWatch logs line-by-line in the console: download them as JSON files and analyse programmatically.

---

### LLM-as-Judge Evaluation

**LLM-as-judge** is a technique where a separate LLM call assesses whether the generated answer is supported by the retrieved context, acting as an automated evaluator in the absence of ground-truth answers.

**Why it matters**

Open-ended queries have no single correct answer to compare against. Traditional accuracy metrics require labeled ground truth. LLM-as-judge makes continuous evaluation possible without human annotation for every query.

**Walkthrough**

Since there is no ground-truth answer for open-ended queries, faithfulness and related metrics are computed using LLM-as-judge. A framework such as **RAGAS** automates this evaluation. RAGAS takes the user query, the retrieved chunks, and the generated answer, then scores faithfulness, context precision, and context coverage automatically.

The deployed demo ran LLM-as-judge evaluation on every query and logged the scores to CloudWatch alongside the interaction data.

**Common mistakes**

- Assuming traditional accuracy metrics apply to open-ended RAG responses: you need LLM-as-judge or similar approaches because there is no labeled ground truth.
- Skipping the evaluation layer for the deployed version: the demo explicitly showed a version without observability (older) and one with CloudWatch metrics and LLM-as-judge (current).

---

### Debugging a Production RAG Failure

A structured debugging loop prevents teams from guessing at the root cause when answer quality drops.

**Why it matters**

Answer quality problems in RAG can have multiple causes — retrieval failure, chunking logic issues, out-of-domain queries, or factual inaccuracy in the LLM response. Structured debugging isolates the cause before a code change is attempted.

**Walkthrough**

When evaluation metrics drop below threshold, follow these steps:

1. Pull recent logs from CloudWatch.
2. Identify queries where answers scored poorly (low faithfulness, low context precision).
3. Compare the user query, retrieved chunks, and generated answer for those queries.
4. Diagnose the root cause: retrieval failure, chunking logic issue, out-of-domain query, or factual inaccuracy.
5. Fix the code in the repository.
6. Rebuild the Docker image, push to ECR with a new tag, and redeploy via Fargate.

**Common mistakes**

- Changing chunking parameters or embedding models without first inspecting the logs. Always pull logs first: they show the exact query, retrieved chunks, and answer that caused the failure.
- Deploying a fix without updating the Docker image tag in ECR: the running containers may continue serving the old image.

---

### Interview Guidance — Describing a Production RAG Project

Knowing how to position a RAG project correctly in an interview separates candidates who built real production systems from those who followed a tutorial.

**Why it matters**

Interviewers know the tutorial stack (FAISS, ChromaDB, small demo datasets). Using production vocabulary signals genuine engineering experience.

**Walkthrough**

The seven key differentiators from a tutorial project are:

1. **Data ingestion pipeline** — a separate cron job process that keeps the vector store current. Mention chunking strategy, metadata extraction, deduplication, and deletion of stale documents.
2. **Production vector store** — Milvus, PGVector, Elasticsearch, OpenSearch, Azure AI Search, or Neo4j. Avoid FAISS or ChromaDB in a production story.
3. **Embedding model** — use a model appropriate to scale. `all-MiniLM-L6-v2` is for demos; a stronger model is expected in production.
4. **Reranking** — retrieve top 10–50 candidates, then rerank to top 3–5.
5. **Observability layer** — CloudWatch (or equivalent) tracking both operational metrics and answer-quality metrics (faithfulness, context precision, context coverage).
6. **Deployment stack** — Docker → ECR → Fargate/ECS → ALB, with CloudFormation for infrastructure management.
7. **Debugging loop** — explain how you would identify and fix a drop in answer quality using logs and metrics.

**Common mistakes**

- Describing FAISS as the vector store in a production story: this immediately signals a tutorial project.
- Describing `all-MiniLM-L6-v2` as the production embedding model: use a stronger model for production; this model is appropriate only for demos and notebooks.

---

## 3. Key Takeaways

- PySpark outperforms sequential Python at scale, but sequential wins for small datasets where startup overhead is not amortised. Measure initialisation latency and throughput separately, not just total elapsed time.
- A production RAG system needs a scheduled ingestion pipeline (fetch → dedup → chunk → embed → upsert) running as a cron job, independent of the user-facing query service.
- Deploying on AWS means Docker → ECR → Fargate behind an Application Load Balancer, with CloudFormation managing the full stack as infrastructure-as-code.
- Observability needs two metric categories: operational (CPU, memory, latency, error rates) and answer-quality (faithfulness, context precision, context coverage). LLM-as-judge via RAGAS automates evaluation without labeled ground truth.
- In an interview, the differentiators are: production vector store (not FAISS/ChromaDB), a dedicated ingestion pipeline, reranking, and an observability layer with a structured debugging loop.

**Mental model:** A production RAG system is a factory with three departments — an ingestion floor, a retrieval-and-generation floor, and an observability control room. Each runs independently; the control room alerts the team when either floor's quality drops.

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