# 58. FastAPI Deployment - Suman - 13 Mar 2026

# FastAPI Deployment

## In-Class Resources: [Click Here](https://coding-platform.s3.amazonaws.com/dev/lms/tickets/5b142fde-9c1e-4be5-813f-09942a87e78b/kYgMktfFeHElcWVx.zip)

**Program:** Vishlesan i-Hub IIT Patna x Masai School — AIM (AI & Machine Learning)

**Topics:** FastAPI, Inference Endpoints, Async Queues, Caching, Production Deployment

---

## Session Structure

- **Part 1:** FastAPI Fundamentals (30 min)
- **Part 2:** ML Inference Endpoints (30 min)
- **Part 3:** Async Programming & Event Loop (30 min)
- **Part 4:** Task Queues with Celery (35 min)
- **Part 5:** Caching Strategies with Redis (30 min)
- **Part 6:** Production Deployment & Monitoring (25 min)

---

# PART 1: FASTAPI FUNDAMENTALS

## 1.1 Why FastAPI for ML?

### Comparison with Other Frameworks

Feature | Flask | Django | FastAPI
Speed | Slow | Slow | Very Fast
Async Support | Limited | Limited | Native
Type Validation | Manual | Manual | Automatic (Pydantic)
API Docs | Manual | Manual | Auto-generated
Learning Curve | Easy | Medium | Easy-Medium
ML Use Case | OK | Overkill | Excellent

### FastAPI Key Features

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Automatic request validation with Pydantic
class PredictionRequest(BaseModel):
    text: str
    max_length: int = 100

@app.post("/predict")
async def predict(request: PredictionRequest):
    # request.text is guaranteed to be string
    # request.max_length defaults to 100
    return {"prediction": "positive"}

```

**Benefits:**

1. **Type safety:** Pydantic validates input automatically
2. **Auto documentation:** Swagger UI at `/docs`, ReDoc at `/redoc`
3. **Async-first:** Built on ASGI (Asynchronous Server Gateway Interface)
4. **Fast:** Built on Starlette and Uvicorn

---

## 1.2 Installation & Setup

### Basic Installation

```bash
# Install FastAPI and ASGI server
pip install fastapi uvicorn[standard]

# For ML deployments
pip install fastapi uvicorn pydantic python-multipart

# Additional dependencies
pip install redis celery aioredis

```

### Minimal FastAPI Application

```python
# main.py
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

```

**Run the server:**

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000

```

**Access:**

- API: [http://localhost:8000/](http://localhost:8000/)
- Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 1.3 Pydantic Models

### Request Validation

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional

class ImageRequest(BaseModel):
    image_url: str
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    return_top_k: int = Field(5, ge=1, le=20)
    
    @validator('image_url')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Must be valid HTTP URL')
        return v

class PredictionResponse(BaseModel):
    predictions: List[dict]
    confidence: float
    processing_time_ms: float

```

### Using Pydantic Models

```python
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: ImageRequest):
    # request is validated automatically
    # If validation fails, FastAPI returns 422 error
    
    start_time = time.time()
    predictions = model.predict(request.image_url)
    
    return PredictionResponse(
        predictions=predictions,
        confidence=max(p['score'] for p in predictions),
        processing_time_ms=(time.time() - start_time) * 1000
    )

```

**Automatic validation:**

- `confidence_threshold` must be between 0.0 and 1.0
- `return_top_k` must be between 1 and 20
- `image_url` must start with http:// or https://
- Response automatically validated against `PredictionResponse`

---

## 1.4 Path Parameters and Query Parameters

### Path Parameters

```python
@app.get("/models/{model_id}")
async def get_model(model_id: str):
    return {"model_id": model_id}

@app.get("/predictions/{task_id}")
async def get_prediction(task_id: str):
    result = redis.get(f"task:{task_id}")
    return {"task_id": task_id, "result": result}

```

### Query Parameters

```python
@app.get("/search")
async def search(
    query: str,
    limit: int = 10,
    offset: int = 0,
    sort_by: Optional[str] = None
):
    # URL: /search?query=cat&limit=5&offset=0&sort_by=relevance
    return {
        "query": query,
        "limit": limit,
        "offset": offset,
        "sort_by": sort_by
    }

```

### Combined Example

```python
@app.get("/users/{user_id}/predictions")
async def get_user_predictions(
    user_id: int,
    limit: int = 10,
    status: Optional[str] = None
):
    # URL: /users/123/predictions?limit=5&status=completed
    return {
        "user_id": user_id,
        "limit": limit,
        "status": status
    }

```

---

# PART 2: ML INFERENCE ENDPOINTS

## 2.1 Model Loading Strategies

### Strategy 1: Load on Startup (Recommended)

```python
import torch
from transformers import pipeline

# Global variable
model = None

@app.on_event("startup")
async def load_model():
    global model
    print("Loading model...")
    model = pipeline("sentiment-analysis")
    print("Model loaded successfully")

@app.post("/predict")
async def predict(text: str):
    # Model already in memory
    result = model(text)
    return result

```

**Pros:**

- Model loaded once
- Fast inference
- Predictable memory usage

**Cons:**

- Startup time increases
- Memory allocated immediately

### Strategy 2: Lazy Loading (Not Recommended)

```python
model = None

def get_model():
    global model
    if model is None:
        model = load_heavy_model()  # 10 seconds!
    return model

@app.post("/predict")
async def predict(text: str):
    # First request takes 10 seconds!
    m = get_model()
    return m(text)

```

**Cons:**

- First request is very slow
- Race condition with concurrent requests
- Unpredictable user experience

### Strategy 3: Dependency Injection (Best Practice)

```python
from fastapi import Depends

class ModelService:
    def __init__(self):
        self.model = None
    
    def load(self):
        if self.model is None:
            self.model = pipeline("sentiment-analysis")
        return self.model

model_service = ModelService()

@app.on_event("startup")
async def startup():
    model_service.load()

def get_model():
    return model_service.model

@app.post("/predict")
async def predict(
    text: str,
    model = Depends(get_model)
):
    result = model(text)
    return result

```

---

## 2.2 File Upload Endpoints

### Single File Upload

```python
from fastapi import File, UploadFile
import numpy as np
from PIL import Image
import io

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    # Read image bytes
    contents = await file.read()
    
    # Convert to PIL Image
    image = Image.open(io.BytesIO(contents))
    
    # Convert to numpy for model
    image_array = np.array(image)
    
    # Run inference
    prediction = model.predict(image_array)
    
    return {
        "filename": file.filename,
        "prediction": prediction
    }

```

### Multiple File Upload

```python
from typing import List

@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    results = []
    
    for file in files:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        prediction = model.predict(np.array(image))
        
        results.append({
            "filename": file.filename,
            "prediction": prediction
        })
    
    return {"results": results}

```

### File Upload with Additional Parameters

```python
from fastapi import Form

@app.post("/predict/image-with-params")
async def predict_with_params(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.5),
    model_version: str = Form("v1")
):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Use specified model version
    model = get_model_version(model_version)
    prediction = model.predict(np.array(image))
    
    # Filter by confidence
    filtered = [p for p in prediction if p['score'] >= confidence_threshold]
    
    return {
        "filename": file.filename,
        "predictions": filtered,
        "model_version": model_version
    }

```

---

## 2.3 Error Handling

### HTTP Exception

```python
from fastapi import HTTPException

@app.post("/predict")
async def predict(text: str):
    if not text or len(text) < 3:
        raise HTTPException(
            status_code=400,
            detail="Text must be at least 3 characters"
        )
    
    try:
        result = model(text)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model inference failed: {str(e)}"
        )

```

### Custom Exception Handler

```python
from fastapi import Request
from fastapi.responses import JSONResponse

class ModelInferenceError(Exception):
    def __init__(self, message: str):
        self.message = message

@app.exception_handler(ModelInferenceError)
async def model_error_handler(request: Request, exc: ModelInferenceError):
    return JSONResponse(
        status_code=500,
        content={
            "error": "model_inference_failed",
            "message": exc.message,
            "path": request.url.path
        }
    )

@app.post("/predict")
async def predict(text: str):
    try:
        result = model(text)
        return result
    except Exception as e:
        raise ModelInferenceError(f"Failed to process: {str(e)}")

```

### Validation Error Customization

```python
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_failed",
            "details": exc.errors(),
            "body": exc.body
        }
    )

```

---

# PART 3: ASYNC PROGRAMMING & EVENT LOOP

## 3.1 Sync vs Async Fundamentals

### Synchronous (Blocking)

```python
import time

def slow_operation():
    time.sleep(2)  # Blocks entire thread
    return "done"

@app.get("/sync")
def sync_endpoint():
    result = slow_operation()  # Blocks for 2 seconds
    return {"result": result}

# Problem: While sleeping, server can't handle other requests
# 10 concurrent requests = 20 seconds total (sequential)

```

### Asynchronous (Non-blocking)

```python
import asyncio

async def slow_operation_async():
    await asyncio.sleep(2)  # Releases control during wait
    return "done"

@app.get("/async")
async def async_endpoint():
    result = await slow_operation_async()  # Yields to event loop
    return {"result": result}

# Benefit: While awaiting, server can handle other requests
# 10 concurrent requests = ~2 seconds total (parallel I/O)

```

### The Event Loop

```
Event Loop:
┌─────────────────────────────────────┐
│  1. Request A arrives               │
│  2. Start processing A              │
│  3. A awaits I/O (database query)   │
│  4. Event loop switches to Request B│
│  5. Start processing B              │
│  6. B awaits I/O (file read)        │
│  7. Event loop switches to Request C│
│  8. Database query for A completes  │
│  9. Event loop returns to A         │
│ 10. A sends response                │
└─────────────────────────────────────┘

```

---

## 3.2 When to Use Async

### Use Async For (I/O-bound):

```python
# Database queries
@app.get("/user/{user_id}")
async def get_user(user_id: int):
    user = await db.fetch_one(f"SELECT * FROM users WHERE id={user_id}")
    return user

# API calls
@app.get("/external-data")
async def fetch_external():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/data")
    return response.json()

# File operations
@app.post("/upload")
async def upload_file(file: UploadFile):
    contents = await file.read()  # Async file read
    await save_to_storage(contents)
    return {"status": "uploaded"}

# Redis/Cache operations
@app.get("/cached/{key}")
async def get_cached(key: str):
    value = await redis.get(key)
    return {"key": key, "value": value}

```

### DON'T Use Async For (CPU-bound):

```python
# ML inference (CPU-intensive)
@app.post("/predict")
async def predict(text: str):
    # WRONG: model.predict() is CPU-bound, not I/O
    # Using async here doesn't help
    result = model.predict(text)  # Blocks event loop!
    return result

# Image processing (CPU-intensive)
@app.post("/process-image")
async def process(file: UploadFile):
    image = await file.read()  # OK: I/O
    processed = heavy_image_processing(image)  # BAD: CPU-bound, blocks
    return processed

```

### Solution: Offload CPU-bound to Background

```python
from fastapi import BackgroundTasks

@app.post("/predict")
async def predict(text: str, background_tasks: BackgroundTasks):
    # Queue CPU-intensive work
    task_id = generate_task_id()
    background_tasks.add_task(run_inference, task_id, text)
    
    return {"task_id": task_id, "status": "queued"}

def run_inference(task_id: str, text: str):
    # Runs in separate thread pool
    result = model.predict(text)
    store_result(task_id, result)

```

---

## 3.3 Async Best Practices

### Always Await Async Functions

```python
# WRONG
async def get_data():
    result = fetch_from_api()  # Missing await!
    return result  # Returns coroutine, not data

# RIGHT
async def get_data():
    result = await fetch_from_api()
    return result

```

### Don't Mix Sync and Async

```python
# WRONG
@app.get("/mixed")
async def mixed_endpoint():
    # sync_function blocks the event loop!
    result = sync_blocking_function()
    return result

# RIGHT: Run sync in thread pool
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=10)

@app.get("/proper")
async def proper_endpoint():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        executor,
        sync_blocking_function
    )
    return result

```

### Async Context Managers

```python
# Database connection
@app.get("/query")
async def query_db():
    async with database.connection() as conn:
        result = await conn.fetch("SELECT * FROM users")
    return result

# HTTP client
@app.get("/external")
async def call_external():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com/data")
    return response.json()

```

---

# PART 4: TASK QUEUES WITH CELERY

## 4.1 Why Task Queues?

### Problem: Long-Running Inference

```python
# BAD: Synchronous, blocks for 30 seconds
@app.post("/generate-report")
async def generate_report(data: dict):
    report = generate_ml_report(data)  # Takes 30 seconds!
    return report

# Issues:
# 1. User waits 30 seconds
# 2. Request might timeout (usually 30s)
# 3. Can't handle concurrent requests efficiently

```

### Solution: Queue the Task

```python
# GOOD: Queue and return immediately
@app.post("/generate-report")
async def generate_report(data: dict):
    task = generate_ml_report_task.delay(data)
    return {
        "task_id": task.id,
        "status": "queued",
        "status_url": f"/tasks/{task.id}"
    }

# Returns in 10ms, processing happens in background

```

---

## 4.2 Celery Setup

### Installation

```bash
pip install celery redis

```

### Celery Configuration

```python
# celery_app.py
from celery import Celery

celery_app = Celery(
    'ml_tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max
    worker_prefetch_multiplier=1,  # Disable prefetching
)

```

### Define Tasks

```python
# tasks.py
from celery_app import celery_app
import time

@celery_app.task(bind=True)
def run_inference(self, image_url: str):
    """
    Long-running ML inference task
    """
    try:
        # Update task state
        self.update_state(
            state='PROCESSING',
            meta={'progress': 0, 'status': 'Loading model'}
        )
        
        # Load model
        model = load_model()
        self.update_state(
            state='PROCESSING',
            meta={'progress': 30, 'status': 'Downloading image'}
        )
        
        # Download image
        image = download_image(image_url)
        self.update_state(
            state='PROCESSING',
            meta={'progress': 60, 'status': 'Running inference'}
        )
        
        # Run inference
        result = model.predict(image)
        
        return {
            'prediction': result,
            'image_url': image_url,
            'completed_at': time.time()
        }
        
    except Exception as e:
        self.update_state(
            state='FAILURE',
            meta={'error': str(e)}
        )
        raise

```

### Integrate with FastAPI

```python
# main.py
from fastapi import FastAPI
from tasks import run_inference
from celery.result import AsyncResult

app = FastAPI()

@app.post("/predict/async")
async def predict_async(image_url: str):
    # Queue the task
    task = run_inference.delay(image_url)
    
    return {
        "task_id": task.id,
        "status": "queued",
        "status_url": f"/tasks/{task.id}"
    }

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    task = AsyncResult(task_id, app=celery_app)
    
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Task is waiting to be processed'
        }
    elif task.state == 'PROCESSING':
        response = {
            'state': task.state,
            'progress': task.info.get('progress', 0),
            'status': task.info.get('status', '')
        }
    elif task.state == 'SUCCESS':
        response = {
            'state': task.state,
            'result': task.result
        }
    elif task.state == 'FAILURE':
        response = {
            'state': task.state,
            'error': str(task.info)
        }
    
    return response

```

---

## 4.3 Running Celery Workers

### Start Redis

```bash
redis-server

```

### Start Celery Worker

```bash
# Single worker
celery -A celery_app worker --loglevel=info

# Multiple workers (for parallelism)
celery -A celery_app worker --loglevel=info --concurrency=4

# With specific queue
celery -A celery_app worker --loglevel=info -Q inference,preprocessing

```

### Monitor with Flower

```bash
pip install flower
celery -A celery_app flower

# Access at http://localhost:5555

```

---

## 4.4 Advanced Celery Patterns

### Task Retries

```python
@celery_app.task(bind=True, max_retries=3)
def unstable_task(self, data):
    try:
        result = potentially_failing_operation(data)
        return result
    except Exception as exc:
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))

```

### Task Routing

```python
# celery_app.py
celery_app.conf.task_routes = {
    'tasks.fast_inference': {'queue': 'fast'},
    'tasks.slow_training': {'queue': 'slow'},
}

# Start workers for specific queues
# Terminal 1: celery -A celery_app worker -Q fast --concurrency=10
# Terminal 2: celery -A celery_app worker -Q slow --concurrency=2

```

### Task Chaining

```python
from celery import chain

# Preprocess → Inference → Postprocess
@app.post("/pipeline")
async def run_pipeline(image_url: str):
    workflow = chain(
        preprocess_task.s(image_url),
        inference_task.s(),
        postprocess_task.s()
    )
    
    result = workflow.apply_async()
    
    return {"task_id": result.id}

```

### Task Groups (Parallel)

```python
from celery import group

@app.post("/batch-predict")
async def batch_predict(image_urls: List[str]):
    # Process all images in parallel
    job = group(inference_task.s(url) for url in image_urls)
    result = job.apply_async()
    
    return {"group_id": result.id}

```

---

# PART 5: CACHING STRATEGIES WITH REDIS

## 5.1 Redis Setup

### Installation

```bash
pip install redis aioredis

```

### Redis Connection

```python
import redis
import aioredis
import json

# Sync Redis
redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True
)

# Async Redis (for FastAPI)
async def get_redis():
    redis = await aioredis.create_redis_pool('redis://localhost')
    return redis

```

---

## 5.2 Basic Caching Pattern

### Simple Cache

```python
import hashlib

@app.post("/predict")
async def predict(text: str):
    # Create cache key from input
    cache_key = f"prediction:{hashlib.md5(text.encode()).hexdigest()}"
    
    # Check cache
    cached_result = redis_client.get(cache_key)
    if cached_result:
        return {
            "prediction": json.loads(cached_result),
            "cached": True
        }
    
    # Cache miss: run inference
    result = model.predict(text)
    
    # Store in cache (expire in 1 hour)
    redis_client.setex(
        cache_key,
        3600,
        json.dumps(result)
    )
    
    return {
        "prediction": result,
        "cached": False
    }

```

### Cache with Version

```python
MODEL_VERSION = "v2.1"

def get_cache_key(input_data: str) -> str:
    data_hash = hashlib.md5(input_data.encode()).hexdigest()
    return f"prediction:{MODEL_VERSION}:{data_hash}"

@app.post("/predict")
async def predict(text: str):
    cache_key = get_cache_key(text)
    
    # Check cache
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Run inference
    result = model.predict(text)
    
    # Cache with version
    redis_client.setex(cache_key, 3600, json.dumps(result))
    
    return result

# When model updates to v2.2, old cache automatically invalid!

```

---

## 5.3 Advanced Caching Patterns

### Cache Aside

```python
async def get_user_predictions(user_id: int):
    cache_key = f"user:{user_id}:predictions"
    
    # Try cache first
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # Cache miss: query database
    predictions = await db.fetch_all(
        f"SELECT * FROM predictions WHERE user_id={user_id}"
    )
    
    # Update cache
    redis_client.setex(
        cache_key,
        300,  # 5 minutes
        json.dumps(predictions)
    )
    
    return predictions

```

### Write-Through Cache

```python
@app.post("/predictions")
async def create_prediction(prediction: dict):
    # Write to database
    prediction_id = await db.insert("predictions", prediction)
    
    # Update cache immediately
    cache_key = f"prediction:{prediction_id}"
    redis_client.setex(
        cache_key,
        3600,
        json.dumps(prediction)
    )
    
    return {"id": prediction_id}

```

### Cache with LRU (Least Recently Used)

```python
from cachetools import LRUCache

# In-memory LRU cache for hot data
hot_cache = LRUCache(maxsize=1000)

@app.get("/predict/{text}")
async def predict_with_lru(text: str):
    # Check hot cache first (fastest)
    if text in hot_cache:
        return hot_cache[text]
    
    # Check Redis (fast)
    cache_key = f"prediction:{hashlib.md5(text.encode()).hexdigest()}"
    cached = redis_client.get(cache_key)
    if cached:
        result = json.loads(cached)
        hot_cache[text] = result  # Promote to hot cache
        return result
    
    # Cache miss: run inference (slow)
    result = model.predict(text)
    
    # Update both caches
    hot_cache[text] = result
    redis_client.setex(cache_key, 3600, json.dumps(result))
    
    return result

```

---

## 5.4 Cache Invalidation

### Time-based Expiration

```python
# Short TTL for frequently changing data
redis_client.setex("stock:AAPL", 5, json.dumps(stock_price))  # 5 seconds

# Long TTL for static data
redis_client.setex("model:weights", 86400, model_data)  # 24 hours

```

### Manual Invalidation

```python
@app.post("/update-model")
async def update_model(new_model_path: str):
    # Load new model
    global model
    model = load_model(new_model_path)
    
    # Invalidate all prediction caches
    keys = redis_client.keys("prediction:*")
    if keys:
        redis_client.delete(*keys)
    
    return {"status": "model updated", "caches_cleared": len(keys)}

```

### Tag-based Invalidation

```python
@app.post("/predict")
async def predict(user_id: int, text: str):
    cache_key = f"prediction:{hashlib.md5(text.encode()).hexdigest()}"
    
    # Tag cache with user
    redis_client.sadd(f"user:{user_id}:caches", cache_key)
    
    # Store prediction
    result = model.predict(text)
    redis_client.setex(cache_key, 3600, json.dumps(result))
    
    return result

@app.delete("/users/{user_id}/cache")
async def clear_user_cache(user_id: int):
    # Get all cache keys for user
    cache_keys = redis_client.smembers(f"user:{user_id}:caches")
    
    # Delete all
    if cache_keys:
        redis_client.delete(*cache_keys)
        redis_client.delete(f"user:{user_id}:caches")
    
    return {"cleared": len(cache_keys)}

```

---

# PART 6: PRODUCTION DEPLOYMENT & MONITORING

## 6.1 Docker Deployment

### Dockerfile for FastAPI

```docker
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    volumes:
      - ./models:/app/models

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  worker:
    build: .
    command: celery -A celery_app worker --loglevel=info --concurrency=4
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    volumes:
      - ./models:/app/models

  flower:
    build: .
    command: celery -A celery_app flower --port=5555
    ports:
      - "5555:5555"
    environment:
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis

```

### Build and Run

```bash
# Build
docker-compose build

# Run
docker-compose up -d

# View logs
docker-compose logs -f api

# Scale workers
docker-compose up -d --scale worker=5

```

---

## 6.2 Health Checks & Monitoring

### Health Check Endpoint

```python
from datetime import datetime

@app.get("/health")
async def health_check():
    # Check model loaded
    model_loaded = model is not None
    
    # Check Redis connection
    try:
        redis_client.ping()
        redis_connected = True
    except:
        redis_connected = False
    
    # Check Celery workers
    from celery_app import celery_app
    stats = celery_app.control.inspect().stats()
    workers_active = len(stats) if stats else 0
    
    healthy = model_loaded and redis_connected and workers_active > 0
    
    return {
        "status": "healthy" if healthy else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {
            "model_loaded": model_loaded,
            "redis_connected": redis_connected,
            "workers_active": workers_active
        }
    }

```

### Request Metrics

```python
import time
from collections import defaultdict

# Simple in-memory metrics
metrics = {
    "requests_total": 0,
    "requests_by_endpoint": defaultdict(int),
    "latency_sum": 0.0,
    "latency_count": 0,
}

@app.middleware("http")
async def metrics_middleware(request, call_next):
    start_time = time.time()
    
    # Track request
    metrics["requests_total"] += 1
    metrics["requests_by_endpoint"][request.url.path] += 1
    
    # Process request
    response = await call_next(request)
    
    # Track latency
    latency = time.time() - start_time
    metrics["latency_sum"] += latency
    metrics["latency_count"] += 1
    
    # Add headers
    response.headers["X-Process-Time"] = str(latency)
    
    return response

@app.get("/metrics")
async def get_metrics():
    avg_latency = (
        metrics["latency_sum"] / metrics["latency_count"]
        if metrics["latency_count"] > 0
        else 0
    )
    
    return {
        "requests_total": metrics["requests_total"],
        "requests_by_endpoint": dict(metrics["requests_by_endpoint"]),
        "average_latency_seconds": avg_latency
    }

```

### Prometheus Integration

```python
from prometheus_client import Counter, Histogram, generate_latest

# Define metrics
request_count = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

request_latency = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['endpoint']
)

@app.middleware("http")
async def prometheus_middleware(request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    # Record metrics
    latency = time.time() - start_time
    request_latency.labels(endpoint=request.url.path).observe(latency)
    request_count.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    return response

@app.get("/metrics/prometheus")
async def prometheus_metrics():
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )

```

---

## 6.3 Rate Limiting

### Simple Rate Limiter

```python
from fastapi import Request, HTTPException
import time

# In-memory rate limit tracker
rate_limits = {}

async def rate_limit_middleware(request: Request, call_next):
    # Get client IP
    client_ip = request.client.host
    
    # Check rate limit
    now = time.time()
    if client_ip in rate_limits:
        requests, window_start = rate_limits[client_ip]
        
        # Reset window if expired (60 seconds)
        if now - window_start > 60:
            rate_limits[client_ip] = [1, now]
        else:
            # Check limit (100 requests per minute)
            if requests >= 100:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Try again in 60 seconds."
                )
            rate_limits[client_ip][0] += 1
    else:
        rate_limits[client_ip] = [1, now]
    
    response = await call_next(request)
    return response

app.middleware("http")(rate_limit_middleware)

```

### Redis-based Rate Limiter

```python
async def check_rate_limit(user_id: str, max_requests: int = 100, window: int = 60):
    """
    Check if user has exceeded rate limit
    Returns (allowed: bool, remaining: int)
    """
    key = f"rate_limit:{user_id}"
    
    # Increment counter
    current = redis_client.incr(key)
    
    # Set expiry on first request
    if current == 1:
        redis_client.expire(key, window)
    
    # Check limit
    if current > max_requests:
        return False, 0
    
    return True, max_requests - current

@app.post("/predict")
async def predict(text: str, user_id: str):
    # Check rate limit
    allowed, remaining = await check_rate_limit(user_id)
    
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Remaining: {remaining}"
        )
    
    # Process request
    result = model.predict(text)
    return result

```

---

## 6.4 Logging Best Practices

### Structured Logging

```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        
        # Add extra fields
        if hasattr(record, 'user_id'):
            log_obj['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_obj['request_id'] = record.request_id
        
        return json.dumps(log_obj)

# Configure logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

@app.post("/predict")
async def predict(text: str, user_id: str = None):
    logger.info(
        "Prediction request received",
        extra={"user_id": user_id, "text_length": len(text)}
    )
    
    try:
        result = model.predict(text)
        logger.info("Prediction successful", extra={"user_id": user_id})
        return result
    except Exception as e:
        logger.error(
            "Prediction failed",
            extra={"user_id": user_id, "error": str(e)}
        )
        raise

```

---

## Summary: Production Checklist

### Deployment Essentials

**✓ API Design:**

- Pydantic models for validation
- Proper error handling
- Health check endpoint
- API documentation (auto-generated)

**✓ Performance:**

- Async endpoints for I/O operations
- Model loaded on startup
- Redis caching implemented
- Cache hit rate >60%

**✓ Scalability:**

- Celery queue for long-running tasks
- Multiple workers configured
- Horizontal scaling ready (Docker)

**✓ Monitoring:**

- Request metrics tracked
- Latency monitoring
- Error logging
- Health checks

**✓ Security:**

- Rate limiting implemented
- Input validation
- Error messages don't leak info
- HTTPS in production

**✓ Reliability:**

- Graceful error handling
- Task retries configured
- Circuit breaker for external APIs
- Database connection pooling

---

**Key Takeaways:**

1. **FastAPI** is ideal for ML: Fast, async-native, auto-validation
2. **Async** for I/O operations, **queues** for CPU-intensive tasks
3. **Caching** can reduce costs by 70%+ (cache hit rate matters!)
4. **Celery** decouples request receipt from processing
5. **Docker** makes deployment consistent and scalable
6. **Monitoring** is not optional — you can't fix what you can't measure

---

**End of Lecture Notes**

**Vishlesan i-Hub IIT Patna × Masai School**

*Build production ML APIs that scale!*

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