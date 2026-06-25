# 12. Lecture Notes: Continuous Delivery for ML - Varun Raste - 7 Nov 2025

## [In-class Notes](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/91b4328f-822d-4435-b9db-54681b8df341/OnsyChRuojFb3GoF.zip)

# Continuous Delivery for ML: Lecture Notes

**Prerequisites:** Understanding of Git workflows (branches, pull requests, merging), familiarity with ML pipelines and reproducibility concepts (MLflow, DVC, config files), basic command line usage, and knowledge of what APIs are (endpoints that receive requests and return responses).

**What you'll be able to do:**

- Create automated CI/CD workflows using GitHub Actions that test and deploy ML models
- Build Docker containers for ML models to ensure consistent environments across development and production
- Design and production deployment strategies that enable safe model updates
- Implement automated model testing and validation in deployment pipelines

---

## 1. Introduction: What is Continuous Delivery for ML and Why Should You Care?

### Core Definition

Continuous Delivery (CD) for ML is the practice of automatically testing, validating, and deploying machine learning models to production environments through automated pipelines, enabling rapid and reliable model updates with minimal manual intervention. Unlike traditional software CD (which deploys code), ML CD deploys models plus their dependencies, handles data validation, monitors model performance, and manages the unique challenges of ML systems where "working code" doesn't guarantee "working predictions." CD extends CI (Continuous Integration) by not just testing code, but actually deploying and production environments automatically or with a single approval click.

### A Simple Analogy

Think of CD for ML like an automated assembly line in a car factory with quality checkpoints. Traditional development is like building cars by hand—each one takes time, is prone to errors, and consistency varies. CD automates the process: when you update your model (like changing the engine design), the pipeline automatically builds it (Docker container), checks quality (model validation), and if everything passes, rolls it out to the main production line (production environment). If issues arise, you can quickly roll back to the previous version. This analogy works for understanding automation and staged deployment, but breaks down when considering ML-specific concerns like data drift and model decay, which don't have parallels in car manufacturing.

### Why This Matters to You

**Problem it solves:** Without CD, ML deployment is painful and risky: manually copying model files to servers, dependency mismatches between development and production ("it works on my laptop"), hours of downtime during updates, no easy rollback when models fail, inconsistent behavior across environments, and manual testing that's slow and incomplete. These problems make teams afraid to update models, leaving stale models in production even when better versions exist.

**What you'll gain:**

- **Speed:** Deploy model updates in minutes instead of days, enabling rapid experimentation and quick response to issues
- **Reliability:** Automated testing catches bugs before production, and containerization eliminates "it works on my laptop" problems

**Real-world context:** Companies like Netflix deploy models thousands of times per day using CD pipelines. Uber uses Docker and Kubernetes to run ML models at massive scale. Airbnb's ML platform automatically validates models before production deployment, preventing costly errors.

---

## 2. The Foundation: Core Concepts Explained

**Note:** Each concept represents a layer in the CD stack. Understanding them individually before seeing how they integrate is essential.

### Concept A: GitHub Actions for CI/CD Automation

**Definition:** GitHub Actions is a CI/CD platform built into GitHub that automatically runs workflows (sequences of jobs and steps) in response to repository events like commits, pull requests, or manual triggers. For ML, workflows can train models, run tests, build Docker images, and deploy to cloud platforms—all defined in YAML files that live in your repository. Unlike external CI tools (Jenkins, CircleCI), GitHub Actions is deeply integrated with GitHub, accessing your code and managing secrets without additional setup.

**Key characteristics:**

- **Event-driven:** Workflows trigger on events (`push`, `pull_request`, `schedule` for cron jobs, `workflow_dispatch` for manual runs)
- **Reusable actions:** Community-built actions (like `checkout` to clone code, `setup-python` to configure Python) simplify workflows
- **Matrix builds:** Run the same job with different configurations (Python 3.8/3.9/3.10, different model types) in parallel for faster testing

**A concrete example:**

```yaml
# .github/workflows/train_model.yml
name: Train and Test Model
on: [push]  # Run on every push

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3  # Clone the repo
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: python train.py
      - run: pytest tests/

```

When you push code, GitHub automatically runs this workflow, training your model and running tests.

**Common confusion:** Beginners think GitHub Actions only runs tests, but it's a full automation platform—it can build Docker images, deploy to cloud platforms, send Slack notifications, create releases, and more. Testing is just one use case. For this lesson, we focus on testing, building, and deployment workflows.

---

### Concept B: Docker for Environment Consistency

**Definition:** Docker is a containerization platform that packages your application (ML model, code, dependencies, Python version, system libraries) into a standardized container image that runs identically anywhere—your laptop, server, or production cloud. A Docker container is like a lightweight virtual machine containing everything needed to run your application, isolated from the host system. The Dockerfile is the recipe describing how to build the container image.

**How it relates to dependency management:** Without Docker, you manage dependencies with requirements.txt and virtual environments, but these don't capture system-level dependencies (like specific versions of CUDA for GPU models). Docker captures everything—Python version, system packages, environment variables—ensuring the exact same environment in development and production.

**Key characteristics:**

- **Immutable:** Once built, a container image doesn't change. If you need updates, you build a new image version
- **Portable:** The same container runs on any system with Docker (Windows, Mac, Linux, cloud platforms)
- **Isolated:** Each container is isolated from others and the host, preventing conflicts between applications

**A concrete example:**

```docker
# Dockerfile
FROM python:3.9-slim  # Start with official Python 3.9 image

WORKDIR /app  # Set working directory inside container

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.pkl train.py ./

CMD ["python", "train.py"]  # Run this command when container starts

```

```bash
# Build the image
docker build -t my-ml-model:v1 .

# Run it anywhere
docker run my-ml-model:v1  # Runs identically on any machine

```

## **Remember:** This is similar to virtual environments (isolation), but differs in that Docker provides full system isolation including OS-level dependencies, making it far more comprehensive than just Python package isolation.

### Concept D: Docker Images vs Containers

**Definition:** A Docker image is a read-only template containing your application and dependencies—think of it as a snapshot or blueprint. A Docker container is a running instance of an image—think of it as a live process. You build an image once (using a Dockerfile), then create many containers from it. Images are stored in registries (like Docker Hub or AWS ECR), and containers run on servers or cloud platforms.

**Key characteristics:**

- **Image:** Static file, versioned (tagged like `my-model:v1.2`), stored in registries, portable across machines
- **Container:** Running process, created from image, has state (logs, temporary files), can be stopped and restarted, many containers can run from one image

**A concrete example:**

```bash
# Build image (once)
docker build -t my-ml-model:v1 .

# Create and run containers (many times, from same image)
docker run -d --name model-instance-1 my-ml-model:v1
docker run -d --name model-instance-2 my-ml-model:v1
docker run -d --name model-instance-3 my-ml-model:v1

# All three containers run independently, from the same image

```

**Remember:** This is similar to the relationship between a class (image) and objects (containers) in programming—one template, many running instances.

---

### Concept E: Workflow Secrets and Environment Variables

**Definition:** Secrets are sensitive values (API keys, database passwords, cloud credentials) that workflows need but shouldn't be committed to code. GitHub Actions stores secrets encrypted and injects them into workflows securely. Environment variables are configuration values passed to containers or scripts, controlling behavior without code changes. Secrets should never appear in logs or code repositories.

**How it relates to configs:** Config files store non-sensitive settings (model hyperparameters, file paths). Secrets store sensitive values (credentials). Both enable configuration without code changes, but secrets have security protections.

**Key characteristics:**

- **GitHub Secrets:** Stored encrypted in GitHub repository settings, injected into workflows as environment variables, never logged or visible in UI
- **Environment variables:** Key-value pairs passed to processes, control container behavior, can be public (like `ENV=production`) or private (from secrets)
- **Best practice:** Never commit secrets to Git, use secret management systems, rotate secrets regularly

**A concrete example:**

```yaml
# .github/workflows/deploy.yml
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Deploy to AWS
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          ENVIRONMENT: production
        run: |
          aws s3 cp model.pkl s3://my-bucket/models/

```

The `secrets.` prefix pulls from GitHub's encrypted secret storage, never exposing the actual values.

**Common confusion:** Beginners sometimes put secrets in config files or code and commit them, creating security vulnerabilities. Always use secret management systems for sensitive values.

---

### How These Concepts Work Together

Think of GitHub Actions as the orchestration conductor, Docker as the standardized packaging format, and production as the quality control checkpoints. In practice: you push code to GitHub, Actions workflows trigger automatically (conductor starts), build Docker images (standardized packages), deploy, run validation tests, and upon success deploy to production (final checkpoint). Environment variables and secrets configure each stage appropriately, and the same Docker image runs in production, ensuring consistency.

---

## 3. Seeing It in Action: Worked Examples

**Tip:** Study these examples carefully before attempting the practice task. Understanding *why* each configuration choice is made and *how* components interact is more important than memorizing syntax.

### Example 1: Basic GitHub Actions Workflow for ML Testing (Simple, Minimal Complexity)

**Scenario:** You have a simple classification model with unit tests for preprocessing functions and integration tests for the model training pipeline. You want tests to run automatically on every commit to catch bugs early.

**Our approach:** Create a GitHub Actions workflow that installs dependencies and runs pytest on every push. This approach makes sense because manual testing before every commit is tedious and easy to forget—automation ensures tests always run.

**Step-by-step solution:**

```yaml
# .github/workflows/test.yml
name: Run ML Pipeline Tests

on:
  push:
    branches: [ main, develop ]  # Run on pushes to these branches
  pull_request:
    branches: [ main ]  # Also run on PRs to main

jobs:
  test:
    runs-on: ubuntu-latest  # Use Ubuntu virtual machine
    
    steps:
      # Step 1: Clone the repository
      - name: Checkout code
        uses: actions/checkout@v3
        # Why: Need the code to test it
      
      # Step 2: Set up Python
      - name: Set up Python 3.9
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
        # Why: Need Python to run our ML code
      
      # Step 3: Install dependencies
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov
        # Why: Need all packages to run tests
      
      # Step 4: Run unit tests
      - name: Run unit tests
        run: |
          pytest tests/unit/ -v --cov=src --cov-report=term-missing
        # Why: Verify preprocessing and utility functions work correctly
      
      # Step 5: Run integration tests
      - name: Run integration tests
        run: |
          pytest tests/integration/ -v
        # Why: Verify the entire pipeline works end-to-end
      
      # Step 6: Check test results
      - name: Report test status
        if: always()  # Run even if tests fail
        run: echo "Test suite completed"

```

**Project structure:**

```
my-ml-project/
├── .github/
│   └── workflows/
│       └── test.yml
├── src/
│   ├── preprocessing.py
│   └── train.py
├── tests/
│   ├── unit/
│   │   └── test_preprocessing.py
│   └── integration/
│       └── test_pipeline.py
├── requirements.txt
└── README.md

```

**Output in GitHub:** When you push code or create a PR, you'll see a green checkmark (tests passed) or red X (tests failed) next to the commit. Click it to see detailed test results.

**What just happened:** GitHub Actions automatically created a fresh Ubuntu virtual machine, installed Python and dependencies, ran all tests, and reported results. This happens on GitHub's servers, not your laptop. If tests fail, the commit is marked as failing, alerting you to fix issues before merging.

**Check your understanding:** Why do we run tests on both `push` and `pull_request` events, not just one?

---

### Example 2: Building and Pushing Docker Images (Adding Complexity)

**Scenario:** You have a trained model that needs to be deployed as an API. You want to containerize it so it runs identically in production, and automatically build new Docker images when the model is updated.

**What's different:** We're building Docker images in CI and pushing them to a container registry (Docker Hub or AWS ECR), making them available for deployment. This adds containerization to our automation pipeline.

**Solution:**

**Step 1: Create Dockerfile**

```docker
# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for layer caching efficiency)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY app.py .

# Expose port for API
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

```

**Step 2: Simple FastAPI application**

```python
# app.py
from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Load model when app starts
model = joblib.load('models/model.pkl')

class PredictionRequest(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(request: PredictionRequest):
    # Convert to numpy array and reshape
    X = np.array(request.features).reshape(1, -1)
    prediction = model.predict(X)
    return {"prediction": int(prediction[0])}

@app.get("/health")
def health():
    return {"status": "healthy"}

```

**Step 3: GitHub Actions workflow to build and push**

```yaml
# .github/workflows/build_and_push.yml
name: Build and Push Docker Image

on:
  push:
    branches: [ main ]
    paths:
      - 'models/**'  # Only trigger when model files change
      - 'Dockerfile'
      - 'requirements.txt'

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
      - name: Checkout code
        uses: actions/checkout@v3
      
      # Step 1: Log in to Docker Hub
      - name: Log in to Docker Hub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
        # Why: Need authentication to push images
      
      # Step 2: Set up Docker Buildx (advanced builder)
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      # Step 3: Build and push image
      - name: Build and push Docker image
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: |
            myusername/ml-model:latest
            myusername/ml-model:${{ github.sha }}
        # Why: Tag with 'latest' and the git commit SHA for versioning
      
      - name: Image built successfully
        run: echo "Image pushed to Docker Hub as myusername/ml-model:latest"

```

**How to set up secrets:**

1. Go to GitHub repository → Settings → Secrets and variables → Actions
2. Add `DOCKER_USERNAME` and `DOCKER_PASSWORD`
3. These are injected securely into the workflow

**Output:** When you push a new model, GitHub Actions builds a Docker image containing your model and API, pushes it to Docker Hub with two tags (`:latest` and `:$SHA` for the git commit), making it ready to deploy.

**Key lesson:** The Docker image is immutable—once built, it won't change. To deploy this exact version, pull `myusername/ml-model:$SHA`. The `:latest` tag always points to the most recent build. Tagging with git SHA enables rollbacks—you know exactly which code version is in each image.

---

### Example 3: Complete CD Pipeline and Production (Real-World Application)

**Background:** A fintech company deploys a fraud detection model as an API. They need to ensure new model versions are thoroughly tested before reaching production, enable fast rollbacks if issues occur, and maintain different configurations for production environments. They've had incidents where bugs in production caused financial losses.

**The challenge:** Manual deployments were slow (4-6 hours including testing and rollback preparation), risky (no systematic validation), and didn't enable quick rollbacks. The team needed automated deployment for testing, automated validation, approval-based production deployment, and rollback capability.

**The approach:** Build a complete CD pipeline:

1. Automated testing on all commits
2. Docker image builds on main branch commits
3. Manual approval gate for production
4. Automatic production deployment after approval
5. Rollback capability by deploying previous image versions

**Implementation:**

**Step 1: Project structure**

```
fraud-detection/
├── .github/
│   └── workflows/
│       ├── test.yml
│       ├── build.yml
│       └── deploy.yml
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   └── api.py
├── tests/
│   ├── unit/
│   └── integration/
├── config/
│   ├── deployment.yaml
│   └── production.yaml
├── deployment/
│   ├── docker-compose.staging.yml
│   └── docker-compose.production.yml
├── Dockerfile
├── requirements.txt
└── validate_model.py

```

**Step 2: Enhanced Dockerfile with environment support**

```docker
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY models/ ./models/
COPY config/ ./config/

# Environment variable to determine which config to use
ENV ENVIRONMENT=staging

# Copy validation script
COPY validate_model.py .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

```

**Step 3: Comprehensive deployment workflow**

```yaml
# .github/workflows/deploy.yml
name: Deploy and Production

on:
  push:
    branches: [ main ]

jobs:
  # Job 1: Run tests
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v
      
  # Job 2: Build Docker image (only if tests pass)
  build:
    needs: test  # Wait for test job to succeed
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Log in to AWS ECR
        uses: aws-actions/amazon-ecr-login@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Build and push to ECR
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          ECR_REPOSITORY: fraud-detection
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG .
          docker tag $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG $ECR_REGISTRY/$ECR_REPOSITORY:latest
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest
          echo "IMAGE_TAG=$IMAGE_TAG" >> $GITHUB_OUTPUT
        id: build-image
  

      
      - name: Deploy to production ECS
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          IMAGE_TAG: ${{ needs.build.outputs.IMAGE_TAG }}
        run: |
          aws ecs update-service \
            --cluster production-cluster \
            --service fraud-detection-service \
            --force-new-deployment \
            --region us-east-1
      
      - name: Verify production deployment
        env:
          PROD_API_URL: https://api.company.com
        run: |
          # Smoke test production
          python validate_model.py --api-url $PROD_API_URL --smoke-test
      
      - name: Send Slack notification
        uses: slackapi/slack-github-action@v1
        with:
          webhook-url: ${{ secrets.SLACK_WEBHOOK }}
          payload: |
            {
              "text": "✅ Fraud detection model deployed to production",
              "blocks": [
                {
                  "type": "section",
                  "text": {
                    "type": "mrkdwn",
                    "text": "*Production Deployment Complete*\nCommit: ${{ github.sha }}"
                  }
                }
              ]
            }

```

**Step 4: Model validation script**

```python
# validate_model.py
import requests
import numpy as np
import argparse
import sys

def validate_model(api_url, smoke_test=False):
    """Validate model performance in deployed environment"""
    
    # Test 1: Health check
    response = requests.get(f"{api_url}/health")
    assert response.status_code == 200, "Health check failed"
    print("✓ Health check passed")
    
    if smoke_test:
        # Just basic functionality test for production
        test_data = {"features": [1.5, 2.3, 0.8, 1.2]}
        response = requests.post(f"{api_url}/predict", json=test_data)
        assert response.status_code == 200, "Prediction endpoint failed"
        print("✓ Smoke test passed")
        return
    
    # Test 2: Prediction correctness
    known_fraud = {"features": [10.5, 15.3, 8.9, 12.1]}  # Known fraud pattern
    known_legit = {"features": [1.2, 0.5, 2.1, 1.8]}     # Known legitimate pattern
    
    fraud_response = requests.post(f"{api_url}/predict", json=known_fraud)
    legit_response = requests.post(f"{api_url}/predict", json=known_legit)
    
    fraud_pred = fraud_response.json()["prediction"]
    legit_pred = legit_response.json()["prediction"]
    
    assert fraud_pred == 1, "Model failed to detect known fraud"
    assert legit_pred == 0, "Model flagged legitimate transaction as fraud"
    print("✓ Prediction correctness passed")
    
    # Test 3: Response time
    import time
    start = time.time()
    requests.post(f"{api_url}/predict", json=known_fraud)
    elapsed = time.time() - start
    
    assert elapsed < 0.5, f"Response time too slow: {elapsed:.2f}s"
    print(f"✓ Response time passed ({elapsed:.3f}s)")
    
    print("\n✅ All validation tests passed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    
    try:
        validate_model(args.api_url, args.smoke_test)
    except AssertionError as e:
        print(f"❌ Validation failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Validation error: {e}")
        sys.exit(1)

```

**Step 5: Environment configurations**

```yaml
# config.yaml
environment: staging
api_settings:
  rate_limit: 1000  # requests per minute
  timeout: 30
  
model:
  path: "s3://company-models/staging/fraud-detection-latest.pkl"
  threshold: 0.5
  
monitoring:
  log_level: DEBUG
  enable_detailed_logging: true

# config/production.yaml  
environment: production
api_settings:
  rate_limit: 10000  # Higher limit for production
  timeout: 10  # Stricter timeout
  
model:
  path: "s3://company-models/production/fraud-detection-latest.pkl"
  threshold: 0.5
  
monitoring:
  log_level: WARNING
  enable_detailed_logging: false

```

---

## 4. Common Pitfalls: What Can Go Wrong and How to Avoid It

**Note:** These aren't just mistakes to avoid—they're learning opportunities to understand the nuances of ML deployment.

### Pitfall 1: Building Different Docker Images for Production

**The Mistake:**

```docker
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
ENV LOG_LEVEL=DEBUG  
CMD ["python", "app.py"]

# Dockerfile.production - uses production settings  
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
ENV LOG_LEVEL=WARNING  # Production-specific
CMD ["python", "app.py"]

```

**The Right Approach:**

```docker
# Single Dockerfile that works in all environments
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Environment variables control behavior, not different Dockerfiles
ENV LOG_LEVEL=${LOG_LEVEL:-INFO}

CMD ["python", "app.py"]

```

```bash
# Build image once
docker build -t my-model:v1 .

# Run in production with production config (same image!)
docker run -e LOG_LEVEL=WARNING -e ENV=production my-model:v1

```

---

**Best practice:** Keep several recent image versions in the registry. Test rollback procedures regularly (quarterly) to ensure they work when needed.

---

### Pitfall 5: Treating ML Deployments Like Software Deployments

**The Mistake:**

```yaml
# Deploy new model like any software update
deploy:
  steps:
    - run: docker push new-model:latest
    - run: kubectl set image deployment/model container=new-model:latest
    # Deployed! But what about model performance validation?

```

You deploy the new model without validating its predictions, monitoring for performance degradation, or comparing against the previous model's behavior.

**Why It's a Problem:** ML models are not regular software. A successfully deployed model doesn't mean a successfully performing model. The model might make poor predictions due to training data issues, feature drift, or bugs in preprocessing that don't cause code errors. You might deploy a model with 60% accuracy when the previous had 85%, and not realize until user complaints roll in.

**The Right Approach:**

```yaml
deploy-staging:
  steps:
    - name: Deploy to staging
      run: ./deploy.sh staging
    
    # ML-specific validation
    - name: Validate model predictions
      run: |
        # Test on known examples
        python test_predictions.py --api-url https://staging-api.company.com
    
    - name: Compare with previous model (shadow mode)
      run: |
        # Send same requests to old and new model, compare outputs
        python compare_models.py \
          --old-model-url https://staging-api-old.company.com \
          --new-model-url https://staging-api.company.com \
          --threshold 0.05  # Fail if accuracy drops > 5%
    
    - name: Load test with realistic data
      run: |
        # Use production-like data patterns
        python load_test.py --data prod_sample.csv --api-url https://staging-api.company.com

deploy-production:
  steps:
    - name: Deploy with gradual rollout
      run: |
        # Canary deployment: route 10% traffic to new model
        kubectl set image deployment/model container=new-model:$SHA
        kubectl scale deployment/model-new --replicas=1
        kubectl scale deployment/model-old --replicas=9
    
    - name: Monitor metrics for 30 minutes
      run: |
        python monitor_model.py \
          --duration 1800 \
          --metrics accuracy,latency,error_rate \
          --alert-threshold 0.05
    
    - name: Full rollout if healthy
      run: |
        # Increase new model to 100% traffic
        kubectl scale deployment/model-new --replicas=10
        kubectl scale deployment/model-old --replicas=0

```

**Model comparison script:**

```python
# compare_models.py
import requests
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

def compare_models(old_url, new_url, test_data, threshold):
    """Compare old and new model predictions"""
    old_predictions = []
    new_predictions = []
    
    for _, row in test_data.iterrows():
        features = row.drop('true_label').tolist()
        
        # Get predictions from both models
        old_pred = requests.post(f"{old_url}/predict", json={"features": features}).json()
        new_pred = requests.post(f"{new_url}/predict", json={"features": features}).json()
        
        old_predictions.append(old_pred['prediction'])
        new_predictions.append(new_pred['prediction'])
    
    true_labels = test_data['true_label'].values
    
    # Calculate accuracy for both
    old_accuracy = accuracy_score(true_labels, old_predictions)
    new_accuracy = accuracy_score(true_labels, new_predictions)
    
    print(f"Old model accuracy: {old_accuracy:.3f}")
    print(f"New model accuracy: {new_accuracy:.3f}")
    print(f"Difference: {new_accuracy - old_accuracy:.3f}")
    
    # Fail if new model is significantly worse
    if new_accuracy < old_accuracy - threshold:
        raise ValueError(f"New model accuracy dropped by {old_accuracy - new_accuracy:.3f}")
    
    print("✓ New model meets performance requirements")

```

**Why This Works:** ML-specific validation ensures the model actually performs well, not just deploys successfully. Shadow mode comparison catches regression in model quality. Canary deployments (gradual rollout) let you monitor real production behavior with minimal risk—if issues arise, only 10% of traffic is affected. This treats ML deployments with the extra validation they require beyond standard software deployment.

---

**If you're stuck:** If GitHub Actions workflows aren't triggering, check the `on:` event configuration and branch names in **Example 1**. If Docker builds fail, review Dockerfile syntax in **Example 2** and check that all copied files exist. If deployments fail, verify secrets are configured in GitHub Settings and check **Pitfall 3** for secret management. If staging/production confusion arises, revisit **Concept C** on environment separation.

---

## 5. Your Turn: Practice & Self-Assessment

### Practice Task (Estimated: 30-35 minutes)

**The Challenge:** You're deploying a house price prediction model as a REST API. Set up a complete CD pipeline with:

1. **GitHub Actions workflow:** Runs tests on every commit, builds Docker image on main branch pushes
2. **Dockerfile:** Containerizes your model API with proper dependency management
3. **Model validation:** Create a validation script that tests predictions and response time
4. **Environment separation:** Different configurations for production

**Specifications:**

- Use FastAPI or Flask for the API (simple `/predict` endpoint)
- GitHub Actions workflow must have separate jobs for test, build, and deploy
- Docker image should be tagged with git SHA
- Validation script must check at least: health endpoint, prediction correctness on known examples, response time
- Create both `config.yaml` and `config/production.yaml`
- Include a manual approval gate before production deployment (use `environment:` in GitHub Actions)

**Hint:** Build incrementally:

1. Create simple FastAPI/Flask app with model prediction endpoint
2. Write basic Dockerfile and test locally (`docker build` and `docker run`)
3. Add tests (pytest for API endpoint testing)
4. Create `.github/workflows/ci.yml` that runs tests
5. Extend workflow to build and push Docker image
6. Add deployment step (can target a local Docker container or docker-compose)
7. Write `validate_model.py` script
8. Add production deployment with manual approval
9. Test the entire workflow by pushing to GitHub

**Extension (optional):**

- Implement model monitoring that logs prediction distribution
- Add Slack/email notifications on deployment success/failure
- Create a rollback workflow that can redeploy previous versions
- Set up a simple Kubernetes deployment or docker-compose for local simulation

---

### Check Your Understanding

Answer these questions to verify you've grasped the key concepts:

1. 
**Explanation question:** Explain why we use the same Docker image and production, rather than building separate images for each environment. What problems does this solve?

2. 
**Application question:** Your GitHub Actions workflow successfully builds and pushes a Docker image, but when you try to deploy it, you get "authentication failed" when pulling from the registry. What are three possible causes and how would you debug each?

3. 
**Error analysis:** Look at this deployment strategy:
`deploy:
  steps:
    - run: docker pull my-model:latest
    - run: docker stop old-model
    - run: docker run --name new-model my-model:latest`

What are two critical problems with this approach, and how would you fix them?

4. 
**Transfer question:** You're deploying a computer vision model that takes 2 seconds per prediction (slow). You need to handle 1000 requests per minute in production. How would you structure your Docker deployment and GitHub Actions workflow to handle this scale? Consider both infrastructure (replicas, load balancing) and CD pipeline concerns (testing at scale, gradual rollout).

**Answers & Explanations:**

1. 
**Single image benefits:** Using the same Docker image and production means you test the exact artifact that will run in production. If you built separate images, subtle differences (different build times, different dependency versions if pinning isn't perfect, different base image patches) could cause bugs to appear in production that weren't present. You control behavioral differences (logging, rate limits) through configuration (environment variables or config files), not different Docker images.

2. 
**Debugging Docker authentication issues:**

**Cause 1: Secrets not configured correctly.** Check GitHub repository settings → Secrets and variables → Actions. Verify `DOCKER_USERNAME` and `DOCKER_PASSWORD` (or equivalent cloud registry credentials) exist and are correct. Debug by printing (non-secret) values: `echo "Using username: ${{ secrets.DOCKER_USERNAME }}"` to verify secret is accessible.
**Cause 2: Wrong registry or image name.** If using AWS ECR/GCS/Azure, verify the registry URL format matches the provider's requirements. Debug by echoing the full image name before pulling: `echo "Pulling $IMAGE_NAME"` and comparing to actual registry.
**Cause 3: Expired credentials or insufficient permissions.** Cloud registry credentials might have expired or lack pull permissions. Debug by testing credentials locally: `docker login <registry>` with the same credentials. Check cloud IAM policies for the service account or user.
All three issues stem from the workflow not having valid credentials to access the registry. Always test authentication separately before debugging the full pipeline.

3. 
**Deployment strategy problems:**

**Problem 1: Downtime during deployment.** The workflow stops the old container before starting the new one, creating a gap where no service is running. Users get errors during this window. Fix with zero-downtime deployment:
`- run: docker pull my-model:latest
- run: docker run -d --name new-model my-model:latest  # Start new container
- run: sleep 10  # Wait for new container to be healthy
- run: docker stop old-model  # Now stop old container
- run: docker rm old-model`

**Problem 2: No rollback capability.** Using `:latest` tag means you can't easily roll back—once pushed, `:latest` points to the new version and you've lost reference to the old version. Fix by tagging with git SHA:
`- run: docker pull my-model:${{ github.sha }}  # Specific version
- run: docker run -d --name new-model my-model:${{ github.sha` 
4. 
**High-throughput deployment strategy:**
`# Dockerfile - optimized for production scale
FROM python:3.9-slim
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy model and code
COPY model/ ./model/
COPY app.py .

# Multiple workers for parallel processing
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", "app:app"]`

`# docker-compose for load balancing
version: '3.8'
services:
  model-1:
    image: my-model:latest
    replicas: 10  # Run 10 containers for load distribution
  
  load-balancer:
    image: nginx:latest
    depends_on:`

---

### Self-Assessment Checklist

You've mastered this topic if you can:

- Create a GitHub Actions workflow that tests, builds, and deploys an ML model without referring to documentation
- Write a Dockerfile that packages an ML model API with proper dependency management and understand image vs container
- Explain to a teammate why environments are critical and how they differ from production
- Debug common deployment issues: authentication failures, image not found, deployment rollback needs
- Design a deployment strategy that handles rollbacks, gradual rollouts, and model-specific validation
- Identify when secrets should use GitHub Secrets vs environment variables vs config files

**If you checked fewer than 5 boxes:** Start by reviewing **Example 1** (basic GitHub Actions) and **Example 2** (Docker basics), practicing each independently before combining them. Then study **Example 3** (complete pipeline) to see integration. If GitHub Actions syntax is confusing, study the official docs' workflow syntax page. If Docker is unclear, practice building and running simple containers before adding ML complexity.

---

## 6. Consolidation: Key Takeaways & Next Steps

### The Essential Ideas

**Core concept recap:**

- **GitHub Actions automates ML workflows:** From testing to building to deploying, automation ensures consistency and catches issues early without manual work
- **Docker guarantees environment consistency:** The same container runs identically across development and production, eliminating "works on my laptop" problems
- **CD enables rapid, safe iteration:** Automated pipelines with proper validation let teams deploy frequently (daily/weekly) instead of quarterly, responding quickly to issues and opportunities

### Mental Model Check

By now, you should think of CD for ML as: An automated assembly line where code commits trigger a series of quality gates (tests, builds, validation) before reaching production, with each gate catching different types of issues, and the ability to quickly roll back the assembly line if a faulty product reaches the end. Unlike manual deployment (slow, risky, inconsistent), CD provides speed, safety, and confidence through automation and staged validation.

### What You Can Now Do

You can now deploy ML models professionally rather than manually copying files to servers. You understand how to automate testing and deployment, eliminate environment inconsistencies with Docker, validate models before production risk, and quickly recover from deployment failures with rollback strategies. These are the foundational skills for ML engineering in production environments, enabling you to deploy models that are reliable, maintainable, and quickly updatable.

### Next Steps

**To deepen this knowledge:**

- Build a complete CD pipeline for a personal ML project, experiencing the full cycle from commit to production
- Explore advanced GitHub Actions features: reusable workflows, composite actions, job matrices
- Learn Docker Compose for multi-container applications (model + database + cache)
- Study Kubernetes basics for orchestrating containers at scale

**To build on this:**

- Learn about advanced deployment strategies: blue-green deployments, A/B testing, shadow mode
- Study model monitoring and observability: logging predictions, detecting drift, alerting on anomalies
- Explore cloud-native ML platforms: AWS SageMaker, Google AI Platform, Azure ML
- Investigate model serving frameworks: TensorFlow Serving, TorchServe, MLflow Models
- Learn about infrastructure as code: Terraform, Pulumi for managing cloud resources

**Additional resources:**

- GitHub Actions documentation: docs.github.com/en/actions (comprehensive workflows guide)
- Docker documentation: docs.docker.com/get-started (official tutorials and best practices)
- Martin Fowler's CD guide: martinfowler.com/articles/continuousIntegration.html (foundational concepts)

---

## Quick Reference Card

Component | Purpose | Key Commands/Patterns
GitHub Actions | Automate test/build/deploy workflows | Define in.github/workflows/*.yml, trigger onpush/pull_request, usejobsandsteps
Docker | Package model + dependencies consistently | docker build -t name:tag .,docker run -e VAR=value name:tag,docker push name:tag
Production | Live environment serving users | Require approval, use versioned images (SHA tags), maintain rollback capability
Secrets | Secure credential management | Store in GitHub Settings → Secrets, access as${{ secrets.NAME }}, never commit to code

**Basic Workflow Pattern:**

```yaml
name: CD Pipeline

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: pytest tests/
  
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: docker build -t my-model:${{ github.sha }} .
      - run: docker push my-model:${{ github.sha }}
  
 

```

---

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