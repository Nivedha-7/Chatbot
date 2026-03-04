import os
import uuid
import traceback
from pathlib import Path
 
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI
 
from ingestion import extract_text, chunk_text
from embeddings import embed_texts
from pgvector_utils import (
    init_db,
    upsert_document,
    insert_chunks,
    list_documents,
    get_chunks_for_doc,
    vector_search_in_doc,
)
 
# -----------------------------
# Load ENV
# -----------------------------
load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)
 
# Debug (optional)
# st.write("ENV file exists:", Path(__file__).with_name(".env").exists())
# st.write("AZURE_OPENAI_ENDPOINT =", os.getenv("AZURE_OPENAI_ENDPOINT"))
# st.write("AZURE_OPENAI_API_VERSION =", os.getenv("AZURE_OPENAI_API_VERSION"))
# st.write("AZURE_OPENAI_CHAT_DEPLOYMENT =", os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"))
# st.write("AZURE_OPENAI_EMBED_DEPLOYMENT =", os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT"))
 
 
def get_aoai_client() -> AzureOpenAI:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
 
    missing = [k for k, v in {
        "AZURE_OPENAI_ENDPOINT": endpoint,
        "AZURE_OPENAI_API_KEY": api_key,
        "AZURE_OPENAI_API_VERSION": api_version,
    }.items() if not v]
 
    if missing:
        raise RuntimeError(f"Missing env values: {', '.join(missing)}")
 
    return AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )
 
 
# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Azure RAG Chatbot", layout="wide")
st.title("Azure RAG Chatbot (Blob + AI Search + PGVector)")  # title can be anything
 
# -----------------------------
# Initialize DB
# -----------------------------
try:
    init_db()
    st.success("Postgres tables created/verified successfully")
except Exception as e:
    st.error("DB init failed")
    st.error(str(e))
    st.stop()
 
# -----------------------------
# Sidebar: Documents
# -----------------------------
st.sidebar.header("📄 Uploaded Documents")
 
docs = []
try:
    docs = list_documents()   # expects list[dict] with keys: id, filename (as your code)
except Exception as e:
    st.sidebar.error("Could not fetch documents")
    st.sidebar.error(str(e))
 
doc_options = ["-- Select a document --"]
doc_map = {}
for d in docs:
    label = f"{d['filename']} | {d['id'][:8]}"
    doc_options.append(label)
    doc_map[label] = d["id"]
 
selected_label = st.sidebar.selectbox("Choose a document for chat", doc_options)
selected_doc_id = doc_map.get(selected_label)
 
if st.sidebar.button("🔄 Refresh document list"):
    st.rerun()
 
# -----------------------------
# Upload + Ingest
# -----------------------------
st.header("Upload document (PDF/DOCX/TXT)")
uploaded_file = st.file_uploader("Drag and drop file here", type=["pdf", "docx", "txt"])
ingest_btn = st.button("Ingest to PGVector")
 
if ingest_btn:
    try:
        if not uploaded_file:
            st.warning("Please upload a file first")
            st.stop()
 
        filename = uploaded_file.name
        file_bytes = uploaded_file.read()
 
        # 1) Extract text
        text = extract_text(filename, file_bytes)
        if not text or len(text.strip()) == 0:
            st.error("No text extracted from document")
            st.stop()
 
        # 2) Chunk
        chunks = chunk_text(text)
        st.write("Chunks created:", len(chunks))
        if not chunks:
            st.error("Chunking produced 0 chunks")
            st.stop()
 
        # 3) Embeddings
        vectors = embed_texts(chunks)
        st.write("Embeddings generated:", len(vectors))
        if not vectors:
            st.error("Embedding generation failed")
            st.stop()
 
        # 4) Store in Postgres
        doc_id = str(uuid.uuid4())
        upsert_document(doc_id, filename)
        insert_chunks(doc_id, chunks, vectors, filename)
 
        st.success("Stored in Postgres successfully ✅")
        st.success(f"Ingestion completed successfully | doc_id = {doc_id}")
        st.info("Now click Refresh document list in sidebar and select this doc for chat.")
 
    except Exception as e:
        st.error("Ingestion failed")
        st.error(str(e))
        traceback.print_exc()
        st.code(traceback.format_exc())
 
st.divider()
 
# -----------------------------
# Preview chunks
# -----------------------------
st.header("📌 Uploaded document preview")
 
if selected_doc_id:
    try:
        st.write("Selected doc_id:", selected_doc_id)
        chunks_preview = get_chunks_for_doc(selected_doc_id, limit=5)
        st.subheader("First 5 chunks stored in PG")
        for i, c in enumerate(chunks_preview, 1):
            st.markdown(f"**Chunk {i}:** {c[:500]}{'...' if len(c) > 500 else ''}")
    except Exception as e:
        st.error("Could not load chunks for this document")
        st.error(str(e))
else:
    st.info("Select a document from the sidebar to preview and chat.")
 
st.divider()
 
# -----------------------------
# Chat (RAG using PGVector)
# -----------------------------
st.header("💬 Chat with the selected PDF (answers only from that PDF)")
 
if not selected_doc_id:
    st.warning("Please select a document from the sidebar first.")
    st.stop()
 
# keep chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
 
# show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
 
question = st.chat_input("Ask a question about the selected document...")
 
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
 
    try:
        # 1) Embed query
        q_vec = embed_texts([question])[0]
 
        # 2) Retrieve top chunks only for selected doc
        top_k = 5
        hits = vector_search_in_doc(selected_doc_id, q_vec, top_k=top_k)
 
        # If your hits are dicts: {"content": "...", ...}
        context = "\n\n".join([h["content"] for h in hits]) if hits else ""
 
        # 3) Call Azure OpenAI (chat completion)
        client = get_aoai_client()
        chat_deploy = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
        if not chat_deploy:
            raise RuntimeError("Missing env value: AZURE_OPENAI_CHAT_DEPLOYMENT")
 
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer ONLY using the given context. "
                           "If the answer is not in the context, say 'Not in the document'."
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
        ]
 
        resp = client.chat.completions.create(
            model=chat_deploy,   # Azure uses deployment name in model field
            messages=messages,
            temperature=0
        )
 
        answer = resp.choices[0].message.content
 
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)
 
        # show sources
        with st.expander("🔎 Sources used (top chunks)"):
            if not hits:
                st.write("No chunks retrieved.")
            else:
                for i, h in enumerate(hits, 1):
                    st.markdown(f"**Source {i} (chunk_index={h.get('chunk_index', 'NA')}):**")
                    st.write(h["content"])
 
    except Exception as e:
        st.error("Chat failed")
        st.error(str(e))
        traceback.print_exc()
        st.code(traceback.format_exc())
 