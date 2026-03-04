# app_langchain.py
import os
import uuid
import io
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
 
from langchain_rag import ingest_text_pg, ask_question_pg, build_faiss_for_doc,ask_question_faiss, get_pg_vectorstore

def load_permanent_docs():
    vs = get_pg_vectorstore()
    
    docs = vs.similarity_search("",k=100) 
    seen={}
    for d in docs:
        doc_id = d.metadata.get("doc_id")
        filename=d.metadata.get('filename')
        if doc_id and filename and doc_id not in seen:
            seen[doc_id]=filename
    return [{"doc_id":k, "filename":v} for k,v in seen.items()]
         
 
load_dotenv(override=True)
 
st.set_page_config(page_title="LangChain Azure RAG", layout="wide")
st.title("LangChain Azure RAG (Azure OpenAI + pgvector)")
 
# ---------------- PDF text extraction ----------------
def extract_pdf_text(file_bytes: bytes) -> str:
    from pypdf import PdfReader
import io
 
def extract_pdf_text(file_bytes: bytes) -> str:
    # 1) Try normal extraction (will be empty for scanned PDFs)
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        texts = []
        for page in reader.pages:
            t = page.extract_text() or ""
            if t.strip():
                texts.append(t)
        normal = "\n".join(texts).strip()
        if normal:
            return normal
    except Exception:
        pass
 
    # 2) OCR fallback (for scanned PDFs)
    from pdf2image import convert_from_bytes
    import pytesseract
 
    images = convert_from_bytes(file_bytes)  # requires Poppler
    ocr_text = []
    for img in images:
        ocr_text.append(pytesseract.image_to_string(img))
    return "\n".join(ocr_text).strip()
 
 
 
# ---------------- Mode handling (Permanent vs Temporary) ----------------
MODE_KEY = "storage_mode"  # 'Permanent' or 'Temporary'
if MODE_KEY not in st.session_state:
    st.session_state[MODE_KEY] = "Permanent"
 
# Separate doc lists for each mode
if "docs_perm" not in st.session_state:
    st.session_state.docs_perm = load_permanent_docs()  # [{doc_id, filename}]
if "docs_temp" not in st.session_state:
    st.session_state.docs_temp = []  # [{doc_id, filename}]
 
# Separate selected doc per mode
if "selected_doc_perm" not in st.session_state:
    st.session_state.selected_doc_perm = None
if "selected_doc_temp" not in st.session_state:
    st.session_state.selected_doc_temp = None
 
# Separate chat history per mode
if "messages_perm" not in st.session_state:
    st.session_state.messages_perm = []
if "messages_temp" not in st.session_state:
    st.session_state.messages_temp = []
 
mode = st.radio(
    "Storage mode",
    ["Permanent", "Temporary"],
    index=0 if st.session_state[MODE_KEY] == "Permanent" else 1,
    horizontal=True,
)
 
# If mode switched: clear chat + reset selection for "fresh" view
if mode != st.session_state[MODE_KEY]:
    st.session_state[MODE_KEY] = mode
 
    if mode == "Permanent":
        st.session_state.messages_perm = []
        st.session_state.selected_doc_perm = None
    else:
        st.session_state.messages_temp = []
        st.session_state.selected_doc_temp = None
 
    st.rerun()
 
 
# ---------------- Sidebar: show docs ONLY for current mode ----------------
with st.sidebar:
    st.header("Uploaded Documents")
 
    if st.button("Refresh list"):
        st.rerun()
 
    docs_list = st.session_state.docs_perm if mode == "Permanent" else st.session_state.docs_temp
 
    if len(docs_list) == 0:
        st.info(f"No documents in {mode} mode yet. Upload a PDF to ingest.")
        selected_doc_id = None
    else:
        options = [f"{d['filename']} | {d['doc_id'][:8]}" for d in docs_list]
        chosen = st.selectbox("Choose a document", options)
 
        selected_doc_id = docs_list[options.index(chosen)]["doc_id"]
 
        # store selection per mode
        if mode == "Permanent":
            st.session_state.selected_doc_perm = selected_doc_id
        else:
            st.session_state.selected_doc_temp = selected_doc_id
 
 
# ---------------- Upload & ingest ----------------
st.subheader(f"Upload & Ingest (PDF) — {mode} mode")
uploaded = st.file_uploader("Upload PDF", type=["pdf"])
 
if st.button("Ingest"):
    if not uploaded:
        st.warning("Upload a PDF first.")
        st.stop()
 
    file_bytes = uploaded.read()
    text = extract_pdf_text(file_bytes)
 
    if not text:
        st.error("No text extracted from PDF.")
        st.stop()
 
    doc_id = str(uuid.uuid4())
 
    # IMPORTANT:
    # - If your backend supports mode, pass it.
    # - If not, this will still run, but both modes would go to the same DB.
    # For now, we keep your existing signature.
    n = ingest_text_pg(doc_id=doc_id, filename=uploaded.name, text=text)
 
    if mode == "Permanent":
        st.session_state.docs_perm.append({"doc_id": doc_id, "filename": uploaded.name})
        st.session_state.selected_doc_perm = doc_id
    else:
        st.session_state.docs_temp.append({"doc_id": doc_id, "filename": uploaded.name})
        st.session_state.selected_doc_temp = doc_id
 
    st.success(f"Ingested {n} chunks ✅ into {mode} list")
    st.info("Now select the doc in the sidebar and ask questions.")
    st.rerun()
 
 
st.divider()
st.subheader("Chat (answers only from selected doc)")
 
if not selected_doc_id:
    st.warning("Select a document from the sidebar.")
    st.stop()
 
# Pick correct chat history for current mode
messages = st.session_state.messages_perm if mode == "Permanent" else st.session_state.messages_temp
 
# Render chat history
for m in messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
 
question = st.chat_input("Ask a question about the selected document...")
if question:
    messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
 
    try:
        answer, sources = ask_question_pg(doc_id=selected_doc_id, question=question, top_k=5)
    except Exception as e:
        st.error(str(e))
        st.stop()
 
    messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
 
    with st.expander("Sources used (metadata)"):
        st.write(sources)
 