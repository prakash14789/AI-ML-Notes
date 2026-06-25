# 24. Lecture notes - Serving Tree Models - Varun Raste - 5 Dec 2025

## [In-class resource](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/322ffd2a-81d7-4b0e-a714-406b0e6bacf1/xlbPisEq3qMSImGa.zip)

# Serving Tree Models: Production Deployment Strategies - Lecture Notes

**Prerequisites:** Understanding of decision trees, random forests, gradient boosting models (XGBoost, LightGBM), basic HTTP/REST concepts, and Python programming.

**What you'll be able to do:**

- Explain the trade-offs between Pickle and ONNX serialization formats for tree models
- Build a REST API stub to serve tree model predictions
- Apply latency testing techniques to measure and optimize model serving performance

---

## 1. Introduction: What is Model Serving and Why Should You Care?

### Core Definition

Model serving is the process of deploying a trained machine learning model into a production environment where it can accept real-time or batch requests and return predictions. It bridges the gap between model development (training in notebooks) and production systems (APIs, web applications, mobile apps). Serving involves serializing the model, creating an interface for requests, managing computational resources, and monitoring performance metrics like latency, throughput, and availability.

### A Simple Analogy

Think of model serving like a restaurant kitchen. You've perfected a recipe (trained model) in your home kitchen (development environment). Now you need to serve it to hundreds of customers daily. You need a way to preserve the recipe consistently (serialization), a waiter system to take orders and deliver food (REST API), and ensure dishes arrive quickly while managing kitchen capacity (latency and throughput optimization). This analogy works for understanding the workflow, but breaks down when considering technical constraints like memory management and concurrent request handling that are unique to software systems.

### Why This Matters to You

**Problem it solves:** Data scientists often build excellent models that never reach production because deployment is complex. Model serving frameworks and best practices enable you to transform experimental models into production-ready services that deliver business value, process millions of predictions daily, and integrate seamlessly with existing software infrastructure.

**What you'll gain:**

- **Production deployment capability:** Transform notebooks into scalable APIs that engineering teams can integrate
- **Performance optimization skills:** Reduce prediction latency from seconds to milliseconds, enabling real-time applications
- **Cross-platform compatibility:** Deploy models across different languages and platforms without retraining

**Real-world context:** Companies like Uber use model serving to predict ETAs in real-time for millions of rides, Netflix serves recommendation models to personalize content for 200+ million users, and fraud detection systems at banks score transactions in under 100ms to approve or decline purchases instantly.

---

## 2. The Foundation: Core Concepts Explained

### Concept A: Model Serialization

**Definition:** Serialization is the process of converting a trained model object (in-memory Python object) into a persistent format that can be saved to disk, transferred across networks, and loaded later for inference without retraining. It preserves the model's learned parameters, architecture, and prediction logic.

**Key characteristics:**

- **Language dependency:** Pickle is Python-specific; ONNX is language-agnostic
- **Format type:** Binary formats that encode model structure and weights
- **Versioning concerns:** Serialized models may break across different library versions

**A concrete example:** After training a Random Forest with `sklearn`, you serialize it with `pickle.dump(model, file)`, creating a `.pkl` file. Later, on a production server, you load it with `pickle.load(file)` to make predictions without retraining.

**Common confusion:** Beginners often think serialization includes the training data or code. The correct understanding is that serialization only captures the model's structure and learned parameters—not the data used to train it or the training code itself.

---

### Concept B: REST API Endpoints

**Definition:** A REST (Representational State Transfer) API endpoint is a URL that accepts HTTP requests (GET, POST) with input data, processes that data through business logic (like model inference), and returns responses in structured formats like JSON. It's the standard interface for applications to communicate over the web.

**How it relates to Model Serialization:** The REST API loads the serialized model into memory when the server starts, then uses that model to generate predictions for each incoming request, creating a stateless service where each request is independent.

**Key characteristics:**

- **HTTP methods:** POST for sending prediction requests with input features
- **JSON payloads:** Input features sent as JSON, predictions returned as JSON
- **Stateless operation:** Each request is independent; the server doesn't remember previous requests

**A concrete example:** A POST request to `https://api.company.com/predict` with JSON `{"features": [5.1, 3.5, 1.4, 0.2]}` returns `{"prediction": "setosa", "probability": 0.98}`.

**Remember:** This is similar to how you interact with any web API (like a weather API), but instead of fetching weather data, you're requesting model predictions based on input features.

---

### How Model Serialization and REST APIs Work Together

Model serialization creates the portable artifact that the REST API loads at startup. The API then holds the deserialized model in memory, uses it to process incoming prediction requests, and returns results. Think of serialization as packaging the recipe, and the REST API as the waiter who takes orders and delivers prepared dishes using that recipe.

---

## 3. Seeing It in Action: Worked Examples

**Tip:** Study these examples carefully before attempting the practice task. Understanding *why* each step is taken is more important than memorizing the steps.

### Example 1: Serializing with Pickle (Simple, Minimal Complexity)

**Scenario:** You've trained a Random Forest classifier for iris flower classification and need to save it for later use in production.

**Our approach:** Use Python's pickle module to serialize the model object to a file, then deserialize it to verify it works correctly. This approach makes sense because it's simple, requires no external dependencies beyond sklearn, and works well for Python-only deployments.

**Step-by-step solution:**

```python
# Step 1: Train a simple model
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import pickle

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)  # Train the model on iris data

# Step 2: Serialize the model to disk
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)  # Save model as binary file

# Step 3: Load the model and verify predictions
with open('rf_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)  # Deserialize from disk

# Verify: predictions should match
original_pred = model.predict([[5.1, 3.5, 1.4, 0.2]])
loaded_pred = loaded_model.predict([[5.1, 3.5, 1.4, 0.2]])
assert (original_pred == loaded_pred).all()  # Confirm consistency

```

**Output:**

```
Assertion passes - model serialization successful
File size: ~2.3 MB for 100-tree Random Forest

```

**What just happened:** We trained a model, converted it to bytes using pickle, saved those bytes to a file, then reconstructed the exact same model from the file. The predictions match perfectly, confirming the serialization preserved all learned parameters.

**Check your understanding:** Why did we use 'wb' and 'rb' modes instead of 'w' and 'r'?

---

### Example 2: Converting to ONNX (One New Element)

**Scenario:** You need to deploy the same Random Forest model to a C++ production service that doesn't support Python.

**What's different:** ONNX (Open Neural Network Exchange) creates a language-agnostic format that can be loaded in C++, Java, JavaScript, or other runtimes, enabling cross-platform deployment.

**Solution:**

```python
# Convert sklearn model to ONNX format
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Define input schema: 4 features (iris dimensions)
initial_type = [('float_input', FloatTensorType([None, 4]))]

# Convert to ONNX format with explicit input types
onnx_model = convert_sklearn(model, initial_types=initial_type)

# Save ONNX model
with open('rf_model.onnx', 'wb') as f:
    f.write(onnx_model.SerializeToString())

# Load and run inference using ONNX Runtime
import onnxruntime as rt
session = rt.InferenceSession('rf_model.onnx')
input_name = session.get_inputs()[0].name
pred = session.run(None, {input_name: [[5.1, 3.5, 1.4, 0.2]]})

```

**Output:**

```
ONNX prediction: [0] (matches sklearn)
File size: ~185 KB (13x smaller than pickle!)

```

**Key lesson:** ONNX models are smaller, faster to load, and platform-independent, but require explicit input type definitions and don't support all sklearn model types or custom preprocessing pipelines as seamlessly as pickle.

---

### Example 3: Building a REST API with Flask (Real-World Application)

**Background:** You work at an e-commerce company that needs to serve product recommendation predictions in real-time. Hundreds of microservices need to query the model, requiring a scalable HTTP API.

**The challenge:** Create a REST endpoint that accepts product features as JSON, runs inference using the serialized model, and returns predictions with sub-100ms latency.

**The approach:** Use Flask to create a lightweight web server, load the pickle model once at startup (not per request), define a POST endpoint that accepts JSON, extracts features, runs model.predict(), and returns JSON responses. This emphasizes loading the model globally to avoid deserialization overhead on every request.

**Why this approach:** Loading the model once at startup means inference requests only pay the prediction cost (milliseconds), not the deserialization cost (seconds). Keeping the service stateless enables horizontal scaling—you can run multiple instances behind a load balancer.

**The outcome:** The API processes 500 requests/second per instance with 15ms average latency. During Black Friday traffic spikes, the team scales from 5 to 50 instances automatically, handling 25,000 req/sec. Performance monitoring shows 99th percentile latency stays below 80ms even under load.

**Caution:** A common mistake is loading the model inside the prediction endpoint function, which adds 2-3 seconds per request. Loading globally at application startup ensures the deserialization cost is paid once, not repeatedly.

---

## 4. Common Pitfalls: What Can Go Wrong and How to Avoid It

**Note:** These aren't just mistakes to avoid—they're learning opportunities to deepen your understanding.

- **The Mistake:** Using pickle models across different Python or sklearn versions

**Why It's a Problem:** Pickle files are version-specific. A model pickled with sklearn 0.24 may fail to load or produce different predictions with sklearn 1.0, causing silent production errors where predictions change unexpectedly after dependency updates.
**The Right Approach:** Use ONNX for version-agnostic deployment, or strictly version-lock all dependencies (Python, sklearn, numpy) using requirements.txt and container images (Docker). Test deserialization in staging environments before production deployment.
**Why This Works:** ONNX uses a standardized computation graph that's independent of library versions, while version-locking ensures the exact environment is reproduced across development and production.

---

- **The Mistake:** Loading the model inside the API route function

**Why It's a Problem:** If you deserialize the model on every request, a simple prediction that should take 5ms becomes 2-3 seconds due to deserialization overhead, making the service unusable for real-time applications.
**The Right Approach:** Load the model once at application startup (global scope or app initialization), keep it in memory, and reuse it for all requests. Use process managers like Gunicorn with multiple workers for parallelism.
**Why This Works:** In-memory model reuse eliminates deserialization latency, and each worker process maintains its own model copy, enabling parallel request processing without locking.

---

- **The Mistake:** Not implementing request batching for high-throughput scenarios

**Why It's a Problem:** Processing one prediction at a time underutilizes modern CPUs and GPUs. Models like XGBoost and neural networks achieve 10-50x higher throughput when predicting on batches of 32-256 samples simultaneously.
**The Right Approach:** Implement dynamic batching—collect incoming requests for 10-50ms, bundle them into a batch, run batch inference, then dispatch individual responses. Libraries like TensorFlow Serving and Ray Serve handle this automatically.
**Why This Works:** Batch predictions leverage vectorized operations and CPU/GPU parallelism, dramatically improving throughput while adding minimal latency (10-50ms batching window vs seconds saved in computation).

**If you're stuck:** Revisit Section 3 Example 3 to understand global model loading patterns, and review batch processing concepts from the ML fundamentals section.

---

## 5. Your Turn: Practice & Self-Assessment

### Practice Task (Estimated: 15-20 minutes)

**The Challenge:** Build a minimal Flask API that serves predictions from a pickled XGBoost model for a binary classification task, measure prediction latency, and identify optimization opportunities.

**Specifications:**

- Create a Flask app with a `/predict` POST endpoint that accepts JSON features
- Load a pre-trained pickled XGBoost model globally at startup
- Return predictions with class label and probability in JSON format
- Implement basic latency logging to track inference time per request
- Test with sample requests and verify responses are correct

**Hint:** Think about how you'd structure the app.py file: model loading happens in global scope before route definitions, the route extracts JSON features into a numpy array matching training shape, and you'll want to use `time.time()` before and after `model.predict()` to measure latency. Consider what happens if a request sends malformed JSON—should the API return a 400 error?

**Extension (optional):** Implement a health check endpoint (`/health`) that returns the model's feature count and version, and add request validation to ensure input has the correct number of features.

---

### Check Your Understanding

Answer these questions to verify you've grasped the key concepts:

1. 
**Explanation question:** Explain in your own words why ONNX models are often smaller than equivalent pickle files, and when this size difference matters in production.

2. 
**Application question:** You're deploying a fraud detection model that needs to score credit card transactions in under 50ms. Would you choose pickle or ONNX, and should you prioritize CPU optimization or implement GPU acceleration? Explain your reasoning.

3. 
**Error analysis:** A Flask API loads a model with `model = pickle.load(open('model.pkl', 'rb'))` inside the route function. Under load testing, latency spikes to 3 seconds per request. What's wrong with this approach, and how would you fix it?

4. 
**Transfer question:** How would you adapt the REST API pattern to serve predictions for a time-series forecasting model that requires the last 30 days of data as input instead of a single feature vector?

**Answers & Explanations:**

1. 
ONNX models are smaller because they use optimized binary protobuf formats that store only the computational graph and weights, eliminating Python-specific overhead that pickle includes (class metadata, module paths). This matters when deploying to edge devices with limited storage, reducing container image sizes in Kubernetes deployments, and speeding up cold-start times in serverless functions (AWS Lambda) where model loading impacts first-request latency.

2. 
Choose pickle for rapid deployment if your entire stack is Python-based, or ONNX if you need cross-language deployment or aggressive optimization. For 50ms latency requirements, focus on CPU optimization first—tree models like XGBoost are highly efficient on CPUs and don't benefit significantly from GPU acceleration (unlike deep learning models). GPU overhead for small batch tree inference often exceeds any computational gains. Prioritize reducing batch size, using compiled ONNX runtimes, and optimizing feature preprocessing.

3. 
The problem is model deserialization happens on every request, adding 2-3 seconds. The fix is to load the model once at module level: `model = pickle.load(open('model.pkl', 'rb'))` in global scope before defining routes. This ensures deserialization happens once at startup, and all requests reuse the in-memory model, reducing latency to milliseconds.

4. 
Modify the `/predict` endpoint to accept a JSON array of 30 daily observations instead of a single feature vector. Reshape the input into the expected format (e.g., `[samples, timesteps, features]` for LSTM models), run inference, and return the forecasted values. Consider caching recent data for users to avoid sending 30 days repeatedly—implement a session-based approach where users send a user_id and only new observations, with the API maintaining a sliding window in Redis or similar cache.

---

### Self-Assessment Checklist

You've mastered this topic if you can:

- Explain the core concept of model serving to someone else in simple terms without looking at notes
- Recognize where pickle vs ONNX is appropriate in new deployment scenarios
- Complete the practice task confidently and explain each decision you made
- Identify and correct common mistakes like in-route model loading without referring back to pitfalls
- Distinguish model serving from model training and understand their different performance constraints
- Build upon this knowledge to learn advanced topics like A/B testing, model versioning, and autoscaling

**If you checked fewer than 5 boxes:** Review Sections 2 and 3 to solidify your understanding of serialization formats and REST API patterns, paying close attention to the worked examples and why each design choice was made.

---

## 6. Consolidation: Key Takeaways & Next Steps

### The Essential Ideas

**Core concept recap:**

- **Serialization enables deployment:** Pickle and ONNX transform trained models into portable artifacts; choose pickle for Python-only simplicity or ONNX for cross-platform and performance-critical deployments.
- **REST APIs bridge models and applications:** Loading models once at startup and exposing prediction endpoints enables scalable, stateless services that integrate with any HTTP client.
- **Latency optimization is critical:** Production models must predict in milliseconds, not seconds; global model loading, batch processing, and profiling are essential techniques.

### Mental Model Check

By now, you should think of model serving as: the engineering discipline of transforming experimental models into production-grade services that meet latency, throughput, and reliability requirements—much like building a restaurant kitchen that can handle dinner rush at scale.

### What You Can Now Do

You've gained the foundational skills to deploy tree models as REST APIs, choose appropriate serialization formats based on deployment constraints, and measure inference performance. These capabilities unlock roles in ML engineering, enable you to collaborate effectively with software engineers, and form the basis for advanced deployment topics like containerization and orchestration.

### Next Steps

**To deepen this knowledge:** Build a complete model serving pipeline: train an XGBoost model on a real dataset, serialize it to both pickle and ONNX, create a Flask API, containerize it with Docker, and deploy to a cloud platform (AWS/GCP/Azure). Implement monitoring with Prometheus to track latency and throughput metrics.

**To build on this:** Explore model versioning and A/B testing frameworks (MLflow, Seldon), learn containerization (Docker, Kubernetes), study advanced serving frameworks (TorchServe, TensorFlow Serving, BentoML), and investigate serverless ML deployment (AWS Lambda, Google Cloud Functions).

**Additional resources:**

- "Designing Machine Learning Systems" by Chip Huyen (Chapter 7 on Model Deployment)
- Flask official documentation for building production APIs with Gunicorn and Nginx

---

**Questions or stuck?** Review the worked examples in Section 3, especially Example 3's REST API implementation. Practice by building simple APIs with different models and serialization formats to gain hands-on confidence.

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