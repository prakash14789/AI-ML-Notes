import os
import streamlit as st
import pandas as pd
import numpy as np
import time

# Import RAG engine components
from rag_engine import (
    NumpyVectorStore, 
    build_notes_index, 
    generate_rag_answer, 
    clean_doc_title
)

# Page configuration for a premium dashboard feel
st.set_page_config(
    page_title="Prakash's AI/ML Notes Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS styling (glassmorphism, clean buttons, custom animations)
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Header style */
    .header-container {
        padding: 1.5rem;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        position: relative;
        overflow: hidden;
    }
    
    .header-container::after {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
        pointer-events: none;
    }
    
    /* Sidebar premium box */
    section[data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Custom cards for metrics */
    .metric-card {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.02);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(0,0,0,0.04);
        border-color: #cbd5e1;
    }
    
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.25rem;
        font-weight: 800;
        color: #1e3c72;
        line-height: 1;
        margin-bottom: 0.25rem;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Source citation link style */
    .source-pill {
        display: inline-block;
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        color: #475569;
        margin-right: 6px;
        margin-bottom: 6px;
        transition: all 0.2s ease;
    }
    
    .source-pill:hover {
        background-color: #e2e8f0;
        color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to get paths
NOTES_DIR = os.path.join(os.path.dirname(__file__), "masai")
CACHE_DIR = os.path.dirname(__file__)

def get_index_path(provider):
    return os.path.join(CACHE_DIR, f"notes_index_{provider}.pkl")

# Initialize Session States
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar Setup
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=70)
    st.title("Settings")
    st.write("Configure your local AI Notes Engine.")
    st.markdown("---")

    # Embedding and LLM Config
    st.subheader("💡 Engine Settings")
    provider = st.selectbox(
        "Embeddings & Indexer Provider",
        options=["local_tfidf", "gemini", "openai"],
        format_func=lambda x: "Local TF-IDF (Offline/Free)" if x == "local_tfidf" else ("Gemini API (Semantic)" if x == "gemini" else "OpenAI API (Semantic)")
    )
    
    llm_provider = st.selectbox(
        "Answer Generation LLM",
        options=["gemini", "openai", "ollama", "offline"],
        format_func=lambda x: (
            "Gemini 2.5 Flash" if x == "gemini" else (
                "GPT-4o Mini" if x == "openai" else (
                    "Ollama Local LLM (Offline)" if x == "ollama" else "Offline Mode (Citations Only)"
                )
            )
        )
    )

    # API Keys & Local Model Settings (conditional visibility)
    api_key = None
    ollama_model = "llama3.2:1b"
    ollama_url = "http://localhost:11434"
    
    if provider == "gemini" or llm_provider == "gemini":
        api_key = st.text_input("Gemini API Key", type="password", help="Grab an API key from Google AI Studio")
    elif provider == "openai" or llm_provider == "openai":
        api_key = st.text_input("OpenAI API Key", type="password", help="Grab an API key from OpenAI Console")
        
    if llm_provider == "ollama":
        ollama_model = st.text_input("Ollama Model Name", value="llama3.2:1b", help="Make sure this model is downloaded locally using 'ollama pull <model_name>'")
        ollama_url = st.text_input("Ollama Server URL", value="http://localhost:11434", help="Default port is 11434")

    st.markdown("---")
    st.subheader("⚙️ Retrieval Parameters")
    top_k = st.slider("Context Chunk Count (Top K)", min_value=1, max_value=10, value=4)
    
    st.markdown("---")
    
    # Index Status & Building
    st.subheader("📂 Index Status")
    index_path = get_index_path(provider)
    index_exists = os.path.exists(index_path)

    if index_exists:
        try:
            # Quick check on number of files
            with open(index_path, 'rb') as f:
                import pickle
                temp_data = pickle.load(f)
                num_chunks = len(temp_data.get("chunks", []))
                unique_files = len(set(c["metadata"]["source_file"] for c in temp_data.get("chunks", [])))
            st.success(f"Index Loaded!\n- {unique_files} notes files\n- {num_chunks} vector chunks")
        except Exception:
            st.warning("Index file corrupted. Please rebuild.")
            index_exists = False
    else:
        st.info("No index found for the selected provider. Please build below.")

    if st.button("🔄 Rebuild Notes Index", use_container_width=True):
        if provider != "local_tfidf" and not api_key:
            st.error("Please input your API Key first to generate semantic embeddings.")
        else:
            with st.spinner("Scanning notes and building vector indexes..."):
                progress_bar = st.progress(0)
                # Walk & count files for accurate progress
                all_md_files = []
                for root, _, files in os.walk(NOTES_DIR):
                    for file in files:
                        if file.lower().endswith('.md') and not file.lower().startswith('readme'):
                            all_md_files.append(os.path.join(root, file))
                
                if not all_md_files:
                    st.error("No markdown files found in the 'masai/' directory.")
                else:
                    progress_step = 1.0 / len(all_md_files)
                    # We override the build function partially to show progress bar updates in UI
                    try:
                        from rag_engine import chunk_markdown_file
                        all_chunks = []
                        for idx, file_path in enumerate(all_md_files):
                            file_chunks = chunk_markdown_file(file_path)
                            all_chunks.extend(file_chunks)
                            progress_bar.progress(min(1.0, (idx + 1) * progress_step))
                        
                        st.info("Embedding chunks via model provider (this may take a few seconds)...")
                        vector_store = NumpyVectorStore(provider=provider)
                        vector_store.add_chunks(all_chunks, api_key=api_key)
                        vector_store.save(index_path)
                        
                        st.success("Successfully built and saved the index!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error rebuilding index: {e}")

# Header
st.markdown("""
<div class="header-container">
    <div style="font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; opacity: 0.85;">
        IIT Patna - Masai AI/ML Course
    </div>
    <h1 style="margin: 0.25rem 0 0.5rem 0; font-size: 2.5rem;">🤖 Study Notes RAG Assistant</h1>
    <p style="margin: 0; opacity: 0.9; font-size: 1.05rem;">
        Semantic search, citations, and chatbot responses compiled directly from Prakash's 92+ study guides.
    </p>
</div>
""", unsafe_allow_html=True)

# Main Application Tabs
tab1, tab2, tab3 = st.tabs(["💬 Chatbot", "📂 Notes Browser", "📊 Analytics & Insights"])

# --- TAB 1: CHATBOT INTERFACE ---
with tab1:
    st.write("Ask any question regarding programming, deep learning, NLP, RAG, MLOps, or deployment models covered in the classes:")

    # Load vector store for session
    vector_store = None
    if index_exists:
        try:
            vector_store = NumpyVectorStore.load(index_path)
        except Exception as e:
            st.error(f"Failed to load notes index: {e}")
    else:
        st.warning("⚠️ **Index not found.** Before chatting, choose a provider in the sidebar and click **Rebuild Notes Index**.")

    # Chat history display
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
                st.markdown("<p style='font-size:0.8rem; font-weight:600; color:#64748b; margin-bottom:5px;'>Sources & Citations:</p>", unsafe_allow_html=True)
                for s in msg["sources"]:
                    meta = s.get("metadata", {})
                    st.markdown(f"<span class='source-pill'>📄 {meta.get('clean_title')} (Sec: {meta.get('header')})</span>", unsafe_allow_html=True)

    # Input area
    if query := st.chat_input("e.g., What are the main differences between bagging and boosting?"):
        # Add query to chat history
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Generate response
        with st.chat_message("assistant"):
            if vector_store is None:
                response_text = "I cannot search the notes because the vector index is not built or failed to load. Please configure the sidebar settings and build the index first."
                st.markdown(response_text)
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
            else:
                with st.spinner("Scanning notes database..."):
                    # Step 1: Vector similarity search
                    try:
                        retrieved = vector_store.similarity_search(query, k=top_k, api_key=api_key)
                    except Exception as e:
                        st.error(f"Search retrieval error: {e}")
                        retrieved = []
                        
                    # Step 2: Answer synthesis
                    if not retrieved:
                        response_text = "I couldn't find any documents or similarity matches matching your query in the study notes."
                        st.markdown(response_text)
                        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                    else:
                            response_text = generate_rag_answer(
                                query, 
                                chunks_only, 
                                provider=llm_provider, 
                                api_key=api_key,
                                ollama_model=ollama_model,
                                ollama_url=ollama_url
                            )
                        except Exception as e:
                            response_text = f"An error occurred during LLM response generation: {e}"
                        
                        st.markdown(response_text)
                        
                        # Source citation expander
                        st.markdown("<p style='font-size:0.8rem; font-weight:600; color:#64748b; margin-bottom:5px; margin-top:15px;'>Sources & Citations:</p>", unsafe_allow_html=True)
                        for chunk, score in retrieved:
                            meta = chunk.get("metadata", {})
                            st.markdown(f"<span class='source-pill' title='Similarity Score: {score:.3f}'>📄 {meta.get('clean_title')} (Sec: {meta.get('header')}) - Score: {score:.2f}</span>", unsafe_allow_html=True)
                        
                        with st.expander("🔍 Show Source Text Snippets"):
                            for i, (chunk, score) in enumerate(retrieved):
                                meta = chunk.get("metadata", {})
                                st.markdown(f"**Source {i+1}: {meta.get('clean_title')}** (Section: *{meta.get('header')}*, Similarity Match: *{score:.2f}*)")
                                st.code(chunk["text"], language="markdown")
                                st.markdown("---")
                                
                        # Save to history
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": response_text,
                            "sources": chunks_only
                        })

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🧹 Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

# --- TAB 2: NOTES BROWSER ---
with tab2:
    st.subheader("🗂️ Explore Class Study Guides")
    st.write("Browse any of your lecture guides directly from within the application, formatted cleanly.")
    
    # Scan notes directory files
    all_files = []
    for root, _, files in os.walk(NOTES_DIR):
        for file in files:
            if file.lower().endswith('.md') and not file.lower().startswith('readme'):
                all_files.append((file, os.path.join(root, file)))
                
    if not all_files:
        st.info("No study guides found in the 'masai/' directory.")
    else:
        # Create clear naming map
        doc_map = {clean_doc_title(f[0]): f[1] for f in all_files}
        sorted_titles = sorted(list(doc_map.keys()))
        
        selected_title = st.selectbox("Select Study Note to view:", options=sorted_titles)
        
        if selected_title:
            note_path = doc_map[selected_title]
            try:
                with open(note_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for line numbers system prefix in standard display
                # If the system displays raw line numbers we can strip them, but let's show standard markdown
                st.markdown(f"### 📂 File: `{os.path.basename(note_path)}`")
                st.markdown("---")
                st.markdown(content)
            except Exception as e:
                st.error(f"Could not load note: {e}")

# --- TAB 3: ANALYTICS & INSIGHTS ---
with tab3:
    st.subheader("📊 Notes Corpus Analytics")
    st.write("High-level breakdown of the knowledge base distribution across weeks and instructors.")

    if index_exists and vector_store is not None:
        # Build pandas dataframe for metadata breakdown
        metadata_list = [c["metadata"] for c in vector_store.chunks]
        df = pd.DataFrame(metadata_list)
        
        if not df.empty:
            total_chunks = len(df)
            unique_notes = df["source_file"].nunique()
            unique_weeks = df[df["week"] > 0]["week"].nunique()
            unique_instructors = df[df["instructor"] != "Unknown"]["instructor"].nunique()

            # Dashboard row metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{unique_notes}</div>
                    <div class="metric-label">Lecture Notes Files</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{total_chunks}</div>
                    <div class="metric-label">Information Chunks</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{unique_weeks}</div>
                    <div class="metric-label">Active Study Weeks</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{unique_instructors}</div>
                    <div class="metric-label">Instructors Covered</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("### 📈 Note Distribution")
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                st.write("**Information Volume by Instructor**")
                # Group by instructor
                inst_counts = df["instructor"].value_counts().reset_index()
                inst_counts.columns = ["Instructor", "Chunk Count"]
                st.bar_chart(inst_counts.set_index("Instructor"))

            with chart_col2:
                st.write("**Active Notes Distribution by Week**")
                week_counts = df[df["week"] > 0]["week"].value_counts().sort_index().reset_index()
                week_counts.columns = ["Week", "Chunk Count"]
                st.line_chart(week_counts.set_index("Week"))
                
            st.markdown("### 🔍 Detailed Corpus Table")
            display_df = df[["clean_title", "instructor", "week", "header"]].drop_duplicates().reset_index(drop=True)
            display_df.columns = ["Note Title", "Instructor", "Course Week", "Example Sections"]
            st.dataframe(display_df, use_container_width=True)
            
        else:
            st.info("No chunk metadata available for analysis. Try rebuilding the index.")
    else:
        st.info("💡 Build the index to see full corpus analytics here.")
