import os
import re
import pickle
import requests
import numpy as np
from typing import List, Dict, Any, Tuple

# Load environment variables from .env file if it exists
if os.path.exists(".env"):
    with open(".env", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")


# Optional imports for API-based models
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    TfidfVectorizer = None
    cosine_similarity = None

class NumpyVectorStore:
    """
    A lightweight, pure-Python and NumPy-based vector store.
    Supports local TF-IDF, Gemini API, and OpenAI API for indexing and similarity search.
    """
    def __init__(self, provider: str = "local_tfidf"):
        self.provider = provider
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: np.ndarray = np.array([])
        
        # TF-IDF Vectorizer if local provider is chosen
        self.tfidf_vectorizer = None
        if provider == "local_tfidf" and TfidfVectorizer is not None:
            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    def add_chunks(self, chunks: List[Dict[str, Any]], api_key: str = None):
        """
        Embeds a list of chunks and stores them.
        Each chunk is a dict: {"text": str, "metadata": dict}
        """
        if not api_key:
            if self.provider == "gemini":
                api_key = os.environ.get("GEMINI_API_KEY")
            elif self.provider == "openai":
                api_key = os.environ.get("OPENAI_API_KEY")
        self.chunks.extend(chunks)
        texts = [c["text"] for c in chunks]

        if not texts:
            return

        if self.provider == "local_tfidf":
            if self.tfidf_vectorizer is None:
                raise ImportError("scikit-learn is required for local TF-IDF. Please run 'pip install scikit-learn'.")
            # Fit and transform
            self.embeddings = self.tfidf_vectorizer.fit_transform(texts)
            
        elif self.provider == "gemini":
            if not api_key:
                raise ValueError("Gemini API key is required.")
            if genai is None:
                raise ImportError("google-genai SDK is required for Gemini embeddings. Run 'pip install google-genai'.")
            
            # Embed via Gemini API
            client = genai.Client(api_key=api_key)
            
            # Process in batches of 100 to avoid request limits
            batch_size = 100
            embeddings_list = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                response = client.models.embed_content(
                    model="text-embedding-004",
                    contents=batch_texts
                )
                # response.embeddings is a list of ContentEmbedding objects
                for emb in response.embeddings:
                    embeddings_list.append(emb.values)
                    
            self.embeddings = np.array(embeddings_list)
            
        elif self.provider == "openai":
            if not api_key:
                raise ValueError("OpenAI API key is required.")
            if OpenAI is None:
                raise ImportError("openai SDK is required for OpenAI embeddings. Run 'pip install openai'.")
                
            client = OpenAI(api_key=api_key)
            batch_size = 100
            embeddings_list = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                response = client.embeddings.create(
                    input=batch_texts,
                    model="text-embedding-3-small"
                )
                for item in response.data:
                    embeddings_list.append(item.embedding)
                    
            self.embeddings = np.array(embeddings_list)

    def similarity_search(self, query: str, k: int = 5, api_key: str = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Searches for the top-k most similar chunks.
        Returns a list of tuples: (chunk, score)
        """
        if not api_key:
            if self.provider == "gemini":
                api_key = os.environ.get("GEMINI_API_KEY")
            elif self.provider == "openai":
                api_key = os.environ.get("OPENAI_API_KEY")
        if not self.chunks or self.embeddings.size == 0:
            return []

        if self.provider == "local_tfidf":
            if self.tfidf_vectorizer is None:
                raise ImportError("scikit-learn is required for local TF-IDF.")
            # Transform query
            query_vector = self.tfidf_vectorizer.transform([query])
            # Compute cosine similarity
            sims = cosine_similarity(query_vector, self.embeddings).flatten()
            top_indices = np.argsort(sims)[::-1][:k]
            return [(self.chunks[idx], float(sims[idx])) for idx in top_indices if sims[idx] > 0.0]

        elif self.provider == "gemini":
            if not api_key:
                raise ValueError("Gemini API key is required.")
            client = genai.Client(api_key=api_key)
            response = client.models.embed_content(
                model="text-embedding-004",
                contents=query
            )
            query_vector = np.array(response.embeddings[0].values)
            
            # Compute Cosine Similarity manually using NumPy
            # sims = dot(embeddings, query) / (norm(embeddings) * norm(query))
            norms = np.linalg.norm(self.embeddings, axis=1)
            query_norm = np.linalg.norm(query_vector)
            
            if query_norm == 0 or np.any(norms == 0):
                return []
                
            sims = np.dot(self.embeddings, query_vector) / (norms * query_norm)
            top_indices = np.argsort(sims)[::-1][:k]
            return [(self.chunks[idx], float(sims[idx])) for idx in top_indices]

        elif self.provider == "openai":
            if not api_key:
                raise ValueError("OpenAI API key is required.")
            client = OpenAI(api_key=api_key)
            response = client.embeddings.create(
                input=[query],
                model="text-embedding-3-small"
            )
            query_vector = np.array(response.data[0].embedding)
            
            norms = np.linalg.norm(self.embeddings, axis=1)
            query_norm = np.linalg.norm(query_vector)
            
            if query_norm == 0 or np.any(norms == 0):
                return []
                
            sims = np.dot(self.embeddings, query_vector) / (norms * query_norm)
            top_indices = np.argsort(sims)[::-1][:k]
            return [(self.chunks[idx], float(sims[idx])) for idx in top_indices]

        return []

    def save(self, filepath: str):
        """Saves the vector store to a pickle file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                "provider": self.provider,
                "chunks": self.chunks,
                "embeddings": self.embeddings,
                "tfidf_vectorizer": self.tfidf_vectorizer
            }, f)

    @classmethod
    def load(cls, filepath: str) -> 'NumpyVectorStore':
        """Loads a vector store from a pickle file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        store = cls(provider=data["provider"])
        store.chunks = data["chunks"]
        store.embeddings = data["embeddings"]
        store.tfidf_vectorizer = data.get("tfidf_vectorizer")
        return store


def clean_doc_title(filename: str) -> str:
    """Cleans note filenames by stripping .md and any trailing Notion/UUID-style 32-character hashes."""
    title = filename[:-3] if filename.lower().endswith('.md') else filename
    # Notion exports suffix a space/dash followed by a 32-character hex ID
    title = re.sub(r'[\s_-]+[a-f0-9]{32}$', '', title, flags=re.IGNORECASE)
    return title.strip()


def parse_week_and_instructor(title: str, first_line: str = "") -> Tuple[int, str]:
    """Tries to extract Week number and Instructor Name from notes titles and first line of file."""
    search_text = f"{title} {first_line}"
    week_match = re.search(r'Week\s*(\d+)', search_text, re.IGNORECASE)
    week_num = int(week_match.group(1)) if week_match else 0
    
    # Common instructors in the files
    instructors = [
        "Sofi Altamsh", "Aashik Arun Bobade", "Varun Raste", "Suman", 
        "Chandan B K", "Chandan", "adarsha khanal", "adarsha", "Nishut", 
        "Dr. Surya Prakash", "Surya Prakash", "Surya"
    ]
    
    instructor = "Unknown"
    for inst in instructors:
        if re.search(re.escape(inst), search_text, re.IGNORECASE):
            # Normalize to clean display names
            if "Surya" in inst:
                instructor = "Dr. Surya Prakash"
            elif "Chandan" in inst:
                instructor = "Chandan B K"
            elif "adarsha" in inst:
                instructor = "Adarsha Khanal"
            elif "Varun" in inst:
                instructor = "Varun Raste"
            else:
                instructor = inst
            break
            
    return week_num, instructor


def chunk_markdown_file(file_path: str, chunk_size_words: int = 150, overlap_words: int = 30) -> List[Dict[str, Any]]:
    """
    Parses a markdown file, splits it into logical word-based chunks,
    and returns a list of dictionaries with text and metadata.
    """
    filename = os.path.basename(file_path)
    clean_title = clean_doc_title(filename)
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

    first_line = lines[0].strip() if lines else ""
    week_num, instructor = parse_week_and_instructor(clean_title, first_line)

    # Clean the lines (remove visual line-number headers if prepended by system, though raw shouldn't have them)
    cleaned_lines = []
    current_header = "Intro"
    
    chunks = []
    current_paragraph = []
    current_word_count = 0
    
    for line in lines:
        stripped = line.strip()
        
        # Keep track of the current header
        if stripped.startswith('#'):
            current_header = stripped.lstrip('#').strip()
            
        cleaned_lines.append((line, current_header))

    # Now let's group words with sliding window chunking
    all_words_with_meta = []
    for line, header in cleaned_lines:
        words = line.split()
        for w in words:
            all_words_with_meta.append((w, header))

    if not all_words_with_meta:
        return []

    # Sliding window chunker preserving words & tracking header
    idx = 0
    total_words = len(all_words_with_meta)
    
    while idx < total_words:
        chunk_slice = all_words_with_meta[idx : idx + chunk_size_words]
        if not chunk_slice:
            break
            
        chunk_text = " ".join([w for w, h in chunk_slice])
        # Use the dominant header in the chunk
        headers = [h for w, h in chunk_slice]
        dominant_header = max(set(headers), key=headers.count) if headers else "Intro"
        
        chunks.append({
            "text": chunk_text,
            "metadata": {
                "source_file": filename,
                "clean_title": clean_title,
                "header": dominant_header,
                "week": week_num,
                "instructor": instructor
            }
        })
        
        if idx + chunk_size_words >= total_words:
            break
            
        idx += (chunk_size_words - overlap_words)

    return chunks


def build_notes_index(notes_dir: str, provider: str = "local_tfidf", api_key: str = None) -> NumpyVectorStore:
    """Walks the notes directory, chunks all markdown files, embeds them, and returns a NumpyVectorStore."""
    all_chunks = []
    for root, dirs, files in os.walk(notes_dir):
        # Exclude hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            if file.lower().endswith('.md') and not file.lower().startswith('readme'):
                file_path = os.path.join(root, file)
                file_chunks = chunk_markdown_file(file_path)
                all_chunks.extend(file_chunks)
                
    vector_store = NumpyVectorStore(provider=provider)
    vector_store.add_chunks(all_chunks, api_key=api_key)
    return vector_store


def generate_rag_answer(
    query: str, 
    retrieved_chunks: List[Dict[str, Any]], 
    provider: str = "gemini", 
    api_key: str = None,
    ollama_model: str = "llama3.2:1b",
    ollama_url: str = "http://localhost:11434"
) -> str:
    """Uses the LLM provider to synthesize an answer based on the retrieved context chunks."""
    if not api_key:
        if provider == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY")
        elif provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
    if not retrieved_chunks:
        return "I could not find any relevant information in your notes to answer this question."

    # Format the context
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks):
        meta = chunk.get("metadata", {})
        context_blocks.append(
            f"--- SOURCE {i+1}: {meta.get('clean_title', 'Unknown')} (Section: {meta.get('header', 'General')}) ---\n"
            f"{chunk['text']}\n"
        )
    context_str = "\n".join(context_blocks)

    system_prompt = (
        "You are an expert AI & Machine Learning teaching assistant. "
        "Your goal is to answer the student's question based strictly on the provided lecture notes.\n\n"
        "Guidelines:\n"
        "1. Be precise, clear, and structure your answer using professional markdown tables, lists, or bold highlights where appropriate.\n"
        "2. Cite your sources clearly using the source names provided in the context (e.g., [SOURCE 1]).\n"
        "3. If the answer cannot be fully deduced from the context, clearly state what the notes cover and answer the question using your general knowledge, but explicitly distinguish what is from the notes vs. general knowledge."
    )

    user_prompt = (
        f"Here are the relevant snippets from the lecture notes:\n\n"
        f"{context_str}\n\n"
        f"Question: {query}\n\n"
        f"Detailed Cited Answer:"
    )

    if provider == "gemini":
        if not api_key:
            return "Please configure your Gemini API Key in the sidebar to generate answers."
        if genai is None:
            return "Error: google-genai library not found. Run pip install google-genai."
            
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.3
            )
        )
        return response.text

    elif provider == "openai":
        if not api_key:
            return "Please configure your OpenAI API Key in the sidebar to generate answers."
        if OpenAI is None:
            return "Error: openai library not found. Run pip install openai."
            
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content

    elif provider == "ollama":
        # Call local Ollama chat API
        url = ollama_url.rstrip('/')
        payload = {
            "model": ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.3
            }
        }
        try:
            response = requests.post(f"{url}/api/chat", json=payload, timeout=45)
            response.raise_for_status()
            res_json = response.json()
            return res_json.get("message", {}).get("content", "Error: Empty response from Ollama.")
        except Exception as e:
            return (
                f"⚠️ **Error connecting to local Ollama server**: {e}\n\n"
                f"Please ensure:\n"
                f"1. The Ollama application is running on your machine (run `ollama serve` or open the Ollama app).\n"
                f"2. You have downloaded the model by running: `ollama pull {ollama_model}` in your terminal.\n"
                f"3. The API URL `{ollama_url}` is correct."
            )

    else:
        # Fallback offline answering (just summaries of retrieved chunks)
        summary = "### Offline Mode (Citations Only)\n"
        summary += "To see AI-generated synthesis, please input your API Key in the sidebar or run Ollama locally. Below are the most relevant snippets found in your notes:\n\n"
        for i, chunk in enumerate(retrieved_chunks):
            meta = chunk.get("metadata", {})
            summary += f"📄 **{meta.get('clean_title')}** (Section: *{meta.get('header')}*, Week: {meta.get('week')})\n"
            summary += f"> *\"{chunk['text'][:300]}...\"*\n\n"
        return summary
