# 44. Mid Project & MLOps Introduction - Suman - 27 Jan 2026

# Lecture Notes: Mid-Project & MLOps Intro

### [PPT File](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/c53e0961-a1f4-47c4-bd31-f9569bd0043d/zfhzSlr3AtmIpkLu.pdf)

### Learning Objectives

By the end of this lecture, you will be able to:

- **Architect** a collaborative RAG pipeline that decouples data, model logic, and frontend development.
- **Implement** Data Version Control (DVC) to track large datasets and model artifacts alongside code.
- **Configure** a Continuous Integration (CI) pipeline to automate linting and testing for machine learning components.
- **Assess** algorithmic bias using formal fairness metrics like Demographic Parity and Equal Opportunity.

---

### Actionable Bridge

You have likely built individual components of a Retrieval-Augmented Generation (RAG) system in isolation. You will use the following concepts when you attempt to merge those components with a partner's work, only to realize that their library versions conflict with yours, or that the dataset you trained on is too large to upload to GitHub.

---

## 1. Collaborative RAG Architecture

### The Problem: The "Notebook Silo"

In early experimentation, it is common to have a single "Monolith.ipynb" file containing data loading, cleaning, vector embedding, and generation logic. While convenient for solo prototyping, this structure is catastrophic for teams. Notebooks do not diff well in version control, and monolithic scripts make it impossible to optimize the retrieval mechanism without risking the generation logic.

### The Solution: Modular Design

To move from a prototype to a production-grade system, you must break the monolith into distinct modules. In a RAG context, this usually implies a separation between the **Indexing Pipeline** (ingesting documents, chunking, embedding) and the **Inference Pipeline** (retrieving context, prompting LLM).

This separation allows for parallel work. One engineer can optimize the chunking strategy to improve retrieval relevance, while another refines the system prompt to reduce hallucinations, provided you agree on the input/output schemas beforehand.

### Implementation: The Service Boundary

In a robust architecture, these components communicate via defined interfaces.

```python
# Interface Definition (Conceptual)

class Retriever:
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """
        Abstracts the vector DB logic.
        Input: User query.
        Output: Top-k relevant document chunks.
        """
        pass

class Generator:
    def generate(self, context: List[Document], query: str) -> str:
        """
        Abstracts the LLM API logic.
        Input: Context chunks and original query.
        Output: Final answer.
        """
        pass

```

By adhering to these interfaces, you can swap the underlying implementation (e.g., changing from ChromaDB to Pinecone, or GPT-3.5 to Llama 2) without breaking your teammate's code.

> 💡 **Pro Tip**: Treat your directory structure as the architecture diagram. A folder named `src/retrieval` and `src/generation` clearly signals separation of concerns, whereas `scripts/run_all.py` signals a monolith.
> 

### Synthesis Point

**Modularization is not just about code organization; it is a risk management strategy.** It isolates errors to specific components and allows teams to upgrade parts of the system (like the vector store) independently of the whole.

---

## 2. Git for ML Teams

### The Problem: The Binary Blob Conflict

Git is optimized for line-based text files. Jupyter Notebooks, however, are JSON files that contain not just code, but also metadata, execution counts, and binary image outputs. If you and a partner both run the same cell in a notebook, the execution counts change. Git sees this as a conflict. Resolving a merge conflict inside a raw JSON file is error-prone and painful.

### The Solution: Branching and Stripping

Effective ML teams use two strategies to mitigate this:

1. **Feature Branching**: Never push directly to `main`. Create branches like `feat/vector-db-setup` or `fix/tokenizer-bug`. This isolates your changes until they are reviewed.
2. **Output Stripping**: Use tools like `nbstripout` to automatically remove cell outputs and execution counts before committing. This ensures Git only tracks the code changes, not the artifacts of your specific run.

### Example: The `.gitignore` for ML

Your repository should exclude heavy or sensitive files. A robust `.gitignore` prevents accidental leaks of API keys or massive model weights.

```
# .gitignore

# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]

# Virtual Environments
venv/
.env

# Jupyter Notebook checkpoints
.ipynb_checkpoints

# Large Data & Models (Managed by DVC, not Git)
data/
models/*.pt
models/*.h5

```

### Trade-offs

Strict branching strategies introduce overhead. You cannot just "save and push." You must commit, push, create a pull request (PR), and merge. However, this friction is intentional; it forces code review and prevents broken code from contaminating the stable `main` branch.

### Synthesis Point

**Git tracks logic, not results.** Configure your environment to strip notebook outputs and ignore large artifacts to keep your version control history clean and diff-able.

---

## 3. Data Version Control (DVC)

### The Problem: The 10GB Dataset

You cannot push a 10GB dataset of PDF documents or a fine-tuned model checkpoint to GitHub. GitHub has strict file size limits (usually 100MB). However, your code depends on specific versions of that data. If you update the code to work with "Dataset V2" but your teammate is still running "Dataset V1," the system will crash or produce garbage results.

### The Solution: Meta-Tracking

**Data Version Control (DVC)** solves this by decoupling the storage of the file from the tracking of the file. DVC stores the actual heavy files in remote storage (S3, Google Drive, Azure Blob Storage) and creates a lightweight metadata file (e.g., `data.csv.dvc`) that lives in Git.

This `*.dvc` file contains a unique hash (usually MD5) of the dataset. When you check out a Git branch, you run `dvc pull`, and DVC looks at the hash in the metadata file, finds the corresponding blob in remote storage, and downloads it to your workspace.

### Implementation: The DVC Workflow

1. 
**Initialize**:
`dvc init`

2. 
**Track Data**:
Instead of `git add data/corpus.json`, you use DVC:
`dvc add data/corpus.json`

This creates `data/corpus.json.dvc`.

3. 
**Link to Git**:
You track the DVC pointer file with Git:
`git add data/corpus.json.dvc .gitignore
git commit -m "Add dataset version 1"`

4. 
**Push Data**:
`dvc push`

**Structure of a `.dvc` file:**

```yaml
outs:
- md5: a805a03429f55694c9255a401d47663d
  path: data/corpus.json

```

Git tracks this text file. DVC tracks the blob `a805a...`.

### Synthesis Point

**DVC connects code versioning to data versioning.** It allows you to time-travel your entire experiment: checking out a Git commit from three weeks ago restores the exact code *and* the exact data used at that time.

---

## 4. Continuous Integration (CI)

### The Problem: "It Works on My Machine"

In a complex RAG system, a change in the document pre-processing logic might silently break the retrieval accuracy, or a syntax error might slip in because you didn't run the full test suite. Relying on every developer to manually run tests before pushing is unreliable.

### The Solution: Automated Gatekeeping

**Continuous Integration (CI)** is the practice of automating the integration of code changes. Every time you push to the repository, a remote server (a "Runner") spins up a fresh environment, installs dependencies, and runs a series of checks. If any check fails, the merge is blocked.

### Implementation: GitHub Actions Pipeline

A typical CI pipeline for an ML project involves three stages: **Linting** (style checks), **Unit Testing** (logic checks), and **Integration Testing** (system checks).

```yaml
# .github/workflows/ci.yml
name: Python application

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: "3.9"
        
    - name: Install dependencies
      run: |
        pip install flake8 pytest
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
        
    - name: Lint with flake8
      run: |
        # Stop the build if there are Python syntax errors or undefined names
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        
    - name: Test with pytest
      run: |
        pytest tests/

```

### Common Pitfalls

- **Env Drift**: If your `requirements.txt` is not pinned (e.g., `pandas` instead of `pandas==1.3.5`), the CI runner might install a newer version than what you have locally, causing the build to fail unexpectedly.
- **Slow Tests**: If your CI takes 20 minutes because it's re-downloading the whole model every time, developers will ignore it. Mock heavy external calls (like OpenAI API requests) during unit tests.

### Synthesis Point

**CI is the heartbeat of a healthy codebase.** It enforces a baseline of quality and prevents "regression"—bugs that reappear after you thought you fixed them.

---

## 5. API Stubs and Contract-First Development

### The Problem: Blocking Dependencies

In a team build, the frontend engineer often needs to build the UI before the ML engineer has finished training the model or setting up the vector database. If the frontend team waits for a working model, development stalls.

### The Solution: Mocking the Interface

An **API Stub** is a dummy implementation of your backend service. It accepts requests and returns valid, hardcoded responses that adhere to the agreed-upon schema (the "contract").

### Implementation: FastAPI Stub

Suppose your RAG system needs an endpoint `/predict`. You can write the API definition immediately:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    text: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/predict", response_model=QueryResponse)
async def predict(request: QueryRequest):
    # TODO: Replace with actual RAG logic later
    # This is the STUB
    return QueryResponse(
        answer="This is a mocked response simulating the LLM output.",
        sources=["doc_1.pdf", "doc_2.txt"]
    )

```

The frontend team can now build their React or Streamlit app against this endpoint. When the real RAG model is ready, you simply replace the body of the `predict` function. The interface (input/output schema) remains unchanged.

### Synthesis Point

**Decouple development timelines using stubs.** Agreeing on the JSON structure of requests and responses early allows parallel development streams to proceed without blocking each other.

---

## 6. Fairness Reflection in ML Systems

### The Problem: Algorithmic Bias

When moving from a notebook to a product, you must evaluate how your model affects different user groups. Models trained on historical data often replicate historical biases. A RAG system used for resume screening might retrieve fewer documents for candidates from underrepresented groups if the embeddings cluster them differently due to vocabulary usage.

### The Solution: Formal Fairness Metrics

We cannot rely on "gut feeling" for fairness. We need mathematical definitions. Two common metrics are **Demographic Parity** and **Equal Opportunity**.

### Demographic Parity

This metric requires that the probability of a positive outcome (e.g., a resume being selected) is the same for all groups, regardless of the sensitive attribute AAA (e.g., gender).

P(Y^=1∣A=a)=P(Y^=1∣A=b)P(\hat{Y} = 1 | A = a) = P(\hat{Y} = 1 | A = b)P(Y^=1∣A=a)=P(Y^=1∣A=b)

- **Trade-off**: This can force the model to be imprecise if the base rates of the target variable differ significantly between groups in the ground truth.

### Equal Opportunity

This metric focuses on the True Positive Rate (TPR). It requires that qualified candidates (where Y=1Y=1Y=1) have the same chance of being selected, regardless of group.

P(Y^=1∣Y=1,A=a)=P(Y^=1∣Y=1,A=b)P(\hat{Y} = 1 | Y = 1, A = a) = P(\hat{Y} = 1 | Y = 1, A = b)P(Y^=1∣Y=1,A=a)=P(Y^=1∣Y=1,A=b)

- **Why it matters**: If your RAG system retrieves context correctly for Group A 90% of the time, but only 60% of the time for Group B, your system fails Equal Opportunity, even if the overall accuracy looks high.

### Practical Application: Confusion Matrices

To audit your system, you must compute a confusion matrix *separately* for each demographic group.

Metric | Group A (Majority) | Group B (Minority) | Disparity
Accuracy | 0.92 | 0.89 | 0.03
False Positive Rate | 0.10 | 0.05 | 0.05
False Negative Rate | 0.05 | 0.25 | 0.20

In this example, the high False Negative Rate for Group B indicates the model is disproportionately rejecting qualified candidates from the minority group. This is a critical failure that aggregate accuracy hides.

### Synthesis Point

**Fairness is not a post-processing step; it is an architectural requirement.** You must log sensitive attributes (securely) during validation to calculate these metrics, or you will be flying blind regarding the ethical impact of your deployment.

---

### Key Takeaways

1. **Modularize Early**: Separate your retrieval logic from your generation logic. This allows you to swap components (like upgrading a vector DB) without rewriting the entire application.
2. **Git is for Code, DVC is for Data**: Never commit large datasets or binary model weights to Git. Use DVC to track data lineage using metadata pointers, keeping your repository lightweight.
3. **Automate Quality Control**: Use CI pipelines to enforce linting and testing on every push. This prevents "integration hell" where conflicting changes break the build right before a deadline.
4. **Contract-First Design**: Use API stubs to define the input/output schema between frontend and backend. This unblocks team members and enforces a clear interface.
5. **Audit for Disparity**: Aggregate accuracy is a misleading metric. Always calculate error rates (FPR/FNR) across demographic subgroups to identify and mitigate algorithmic bias.
6. **Resolve Conflicts via Process**: Use branching strategies and tools like `nbstripout` to manage the inherent difficulty of collaborating on Jupyter Notebooks.

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