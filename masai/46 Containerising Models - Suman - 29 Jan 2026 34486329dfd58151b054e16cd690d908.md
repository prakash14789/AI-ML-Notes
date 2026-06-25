# 46. Containerising Models - Suman - 29 Jan 2026

# Containerising Models: Dockerfile Patterns, Dependency Pinning, and Slim Images

# [In-Class Notes](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/cfb15a60-9788-4dbb-8498-a0de54bb1976/mkJ0WNPaozWt03FY.zip)

**Prerequisites:** Basic Docker commands (build, run, push), Python virtual environments, pip package management

**What you'll be able to do:**

- Build production-grade Dockerfiles for ML models using proven patterns
- Implement dependency pinning to ensure reproducible builds
- Choose appropriate base images and optimise container size

---

## 1. Introduction: What is Model Containerisation and Why Should You Care?

### Core Definition

Model containerisation packages your trained ML model, inference code, and all dependencies into a portable Docker image. This image runs identically on your laptop, a CI server, or a Kubernetes cluster with thousands of nodes. Unlike virtual machines, containers share the host kernel, making them lightweight and fast to start.

### A Simple Analogy

Think of a container as a meal prep kit. Everything needed to cook the dish—ingredients, spices, utensils—comes in one box with exact quantities and instructions. You don't need to shop for ingredients or guess measurements. The result is consistent every time, regardless of whose kitchen you're in.

**Limitation:** This analogy breaks down for resource sharing—meal kits are independent, but containers share the host's CPU, memory, and kernel.

### Why This Matters to You

**Problem it solves:** The "dependency hell" where your model works locally but fails in production due to library version mismatches, missing system packages, or Python version differences.

**What you'll gain:**

- Reproducible deployments that work the same everywhere
- Faster iteration cycles with efficient Docker layer caching
- Smaller images that deploy faster and cost less to store

**Real-world context:** Every major ML platform—SageMaker, Vertex AI, Azure ML—runs models in containers. Mastering containerisation is essential for production ML.

---

## 2. The Foundation: Core Concepts Explained

### Concept A: Dockerfile Instruction Order and Layer Caching

**Definition:** Docker builds images by executing Dockerfile instructions sequentially, creating a cached layer for each instruction. When you rebuild, Docker reuses cached layers until it encounters a change, then rebuilds everything after that point.

**Key characteristics:**

- Layers are immutable and stacked on top of each other
- A change in any layer invalidates all subsequent layers
- Layer caching dramatically speeds up iterative development

**A concrete example:**

```docker
COPY requirements.txt .      # Layer 1: Changes rarely
RUN pip install -r requirements.txt  # Layer 2: Cached if requirements unchanged
COPY . .                     # Layer 3: Changes frequently

```

**Common confusion:** Beginners often copy all files first, then install dependencies. This means every code change invalidates the dependency cache, forcing a full reinstall.

---

### Concept B: Dependency Pinning

**Definition:** Dependency pinning means specifying exact versions for every package your application uses, including transitive dependencies (dependencies of your dependencies). This ensures builds are reproducible regardless of when or where they run.

**How it relates to Dockerfile patterns:** Pinned dependencies make Docker layer caching more effective—the pip install layer only rebuilds when you intentionally update versions.

**Key characteristics:**

- Direct dependencies: packages you explicitly import
- Transitive dependencies: packages your dependencies need
- Lock files capture the entire dependency tree at a point in time

**A concrete example:**

```
# Bad: requirements.txt
pandas
scikit-learn

# Good: requirements.txt (pinned)
pandas==2.0.3
scikit-learn==1.3.0
numpy==1.24.3
joblib==1.3.1

```

**Remember:** This is similar to how `package-lock.json` works in Node.js or `Pipfile.lock` in Pipenv—capturing exact versions for reproducibility.

---

### How Dockerfile Patterns and Dependency Pinning Work Together

Proper instruction ordering maximises cache hits, while dependency pinning ensures those cached layers remain valid over time. Together, they create fast, reproducible builds. Change your model code? Rebuild in seconds. Update a dependency? Explicitly and intentionally.

---

## 3. Seeing It in Action: Worked Example

**Tip:** Study this example carefully. Understanding why each decision was made matters more than memorising the syntax.

### Production Dockerfile for an ML Model

**Scenario:** You're deploying a customer churn prediction model. It uses scikit-learn for the model, FastAPI for the REST API, and needs to be small enough for fast Kubernetes deployments.

**Our approach:** Use a multi-stage build with a slim base image. Install dependencies first, copy code second. Run as non-root for security.

**Step-by-step solution:**

```docker
# Stage 1: Builder stage for compiling dependencies
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies (only needed for compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

# Stage 2: Runtime stage (final image)
FROM python:3.10-slim

WORKDIR /app

# Copy pre-built wheels from builder stage
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Create non-root user
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Health check for orchestration
HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
ENTRYPOINT ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

```

**Output:** A Docker image around 200-300MB instead of 1-2GB, with no build tools in the final image.

**What just happened:** The builder stage compiled any packages needing C extensions (like numpy), then the runtime stage copied only the compiled wheels. The gcc compiler never appears in the final image.

**Check your understanding:** Why do we delete `/var/lib/apt/lists/*` after apt-get install?

---

## 4. Common Pitfalls: What Can Go Wrong

**Note:** These mistakes are learning opportunities. Understanding why they fail deepens your knowledge.

- **The Mistake:** Copying all files before installing dependencies

**Why It's a Problem:** Every code change invalidates the pip install layer, causing 5-10 minute rebuilds
**The Right Approach:** `COPY requirements.txt .` then `RUN pip install`, then `COPY . .`
**Why This Works:** Requirements change rarely; code changes constantly. Separate them.

---

- **The Mistake:** Using unpinned dependencies (`pandas` instead of `pandas==2.0.3`)

**Why It's a Problem:** A new pandas release on Tuesday breaks your Wednesday deployment with no code changes
**The Right Approach:** Generate locked requirements with `pip freeze > requirements.txt` or use pip-tools
**Why This Works:** Explicit versions make builds deterministic and debugging possible

---

- **The Mistake:** Running containers as root

**Why It's a Problem:** If an attacker exploits your application, they have root access to the container (and potentially the host)
**The Right Approach:** Create a non-root user with `useradd` and switch to it with `USER`
**Why This Works:** Principle of least privilege—your app doesn't need root, so don't give it root

**If you're stuck:** Revisit the multi-stage build example—it demonstrates all three correct approaches.

---

## 5. Your Turn: Practice & Self-Assessment

### Practice Task (Estimated: 20 minutes)

**The Challenge:** Create a Dockerfile for a sentiment analysis model that uses transformers and FastAPI.

**Specifications:**

- Use `python:3.10-slim` as the base image
- Pin all dependencies in requirements.txt
- Order instructions to maximise layer caching
- Run as a non-root user named `mluser`
- Expose port 8080

**Hint:** Start by listing what changes frequently (your code) vs rarely (dependencies). Put the rarely-changing items earlier in the Dockerfile. For the transformers library, you may need build tools in a separate builder stage.

**Extension (optional):** Add a multi-stage build that uses `python:3.10` (with build tools) as the builder and `python:3.10-slim` as the runtime.

---

### Check Your Understanding

1. **Explanation question:** Why does the order of COPY and RUN instructions matter for build performance?
2. **Application question:** Your Docker build takes 8 minutes but your only change was fixing a typo in a comment. What's likely wrong with your Dockerfile structure?
3. **Error analysis:** What's wrong with this Dockerfile snippet?
`COPY . .
RUN pip install -r requirements.txt`

4. **Transfer question:** How would you apply layer caching principles if you were building a Node.js application instead of Python?

**Answers & Explanations:**

1. Docker caches each layer. If an earlier layer changes, all subsequent layers rebuild. Putting stable content (dependencies) before volatile content (code) maximises cache hits.
2. Your code is likely copied before dependencies are installed. The typo fix invalidated the dependency layer, causing a full reinstall.
3. Copying everything first means any code change invalidates the pip install cache. Fix: `COPY requirements.txt .` first, then `RUN pip install`, then `COPY . .`
4. Same principle: `COPY package.json .` then `RUN npm install` then `COPY . .`—copy the dependency manifest before installing, copy code last.

---

## 6. Consolidation: Key Takeaways & Next Steps

### The Essential Ideas

- **Order matters:** Copy dependency files, install dependencies, then copy code—maximises cache efficiency
- **Pin everything:** Exact versions for all packages prevent "it worked yesterday" failures
- **Go slim:** Slim base images reduce size by 5-10x; multi-stage builds remove build tools from runtime

### Mental Model Check

By now, you should think of a Dockerfile as a sequence of cached checkpoints. Each instruction is a checkpoint Docker can skip to if nothing before it changed. Your job is arranging instructions so Docker can skip as many checkpoints as possible.

### What You Can Now Do

You can build production-grade Docker images that build fast, run securely, and deploy reliably. This skill transfers directly to any ML deployment—whether SageMaker, Kubernetes, or a simple EC2 instance.

### Next Steps

**To deepen this knowledge:** Build Docker images for different ML frameworks (PyTorch, TensorFlow) and compare their size and build times.

**To build on this:** Learn Kubernetes deployments to run your containers at scale with automatic scaling and health management.

**Additional resources:** Docker's official best practices guide for Dockerfile authoring.

---

## Quick Reference Card

Practice | Bad | Good
Instruction order | COPY . . → pip install | COPY requirements.txt → pip install → COPY . .
Dependencies | pandas | pandas==2.0.3
Base image | python:3.10 (900MB) | python:3.10-slim (150MB)
User | root (default) | Non-root user with USER instruction
Build tools | Include in final image | Multi-stage build, only in builder

---

**Questions or stuck?** Review the multi-stage build example—it demonstrates every best practice in one place.

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