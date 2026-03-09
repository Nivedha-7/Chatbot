import uuid
import streamlit as st
from langchain_rag import RAGChatbotService
 
st.set_page_config(page_title="LangChain Azure RAG", layout="wide")
st.title("Chatbot")
 
bot = RAGChatbotService()
 
# ---------------- Session State ----------------
st.session_state.setdefault("mode", "Permanent")
 
st.session_state.setdefault("perm_docs", [])
st.session_state.setdefault("temp_docs", [])
st.session_state.setdefault("temp_stores", {})
 
st.session_state.setdefault("selected_doc_perm", None)
st.session_state.setdefault("selected_doc_temp", None)
 
st.session_state.setdefault("chat_perm", [])
st.session_state.setdefault("chat_temp", [])
 
st.session_state.setdefault("auto_delete_temp", True)
 
# ---------------- Sidebar: storage mode ----------------
with st.sidebar:
    st.header("Storage Mode")
 
    mode = st.radio(
        "Choose mode",
        ["Permanent", "Temporary"],
        index=0 if st.session_state.mode == "Permanent" else 1
    )
 
    if mode != st.session_state.mode:
        st.session_state.mode = mode
 
        # clear uploader widget state when switching modes
        if mode == "Permanent":
            st.session_state.pop("temp_uploader", None)
        else:
            st.session_state.pop("perm_uploader", None)
 
        st.rerun()
 
    if st.session_state.mode == "Temporary":
        st.checkbox("Auto-delete temp doc after answer", key="auto_delete_temp")
 
# ---------------- Load permanent docs automatically ----------------
if st.session_state.mode == "Permanent":
    try:
        st.session_state.perm_docs = bot.permanent_storage.list_documents()
    except Exception as e:
        st.warning(f"Could not load permanent docs: {e}")
 
# ---------------- Sidebar: dropdown + buttons ----------------
with st.sidebar:
    st.subheader("Uploaded Documents")
 
    if st.session_state.mode == "Permanent":
        docs_list = st.session_state.perm_docs
 
        if not docs_list:
            st.info("No permanent docs yet.")
            st.session_state.selected_doc_perm = None
        else:
            options = [f'{d["filename"]} | {d["doc_id"][:8]}' for d in docs_list]
 
            selected_index = 0
            if st.session_state.selected_doc_perm is not None:
                for i, d in enumerate(docs_list):
                    if d["doc_id"] == st.session_state.selected_doc_perm:
                        selected_index = i
                        break
 
            chosen = st.selectbox(
                "Choose permanent document",
                options,
                index=selected_index,
                key="perm_doc_dropdown"
            )
            st.session_state.selected_doc_perm = docs_list[options.index(chosen)]["doc_id"]
 
        c1, c2 = st.columns(2)
 
        with c1:
            if st.button("Refresh", use_container_width=True):
                st.session_state.perm_docs = bot.permanent_storage.list_documents()
                st.rerun()
 
        with c2:
            if st.button("Delete", use_container_width=True):
                if st.session_state.selected_doc_perm is None:
                    st.warning("Select a document first.")
                else:
                    try:
                        bot.permanent_storage.delete_document(st.session_state.selected_doc_perm)
                        st.session_state.perm_docs = bot.permanent_storage.list_documents()
                        st.session_state.selected_doc_perm = None
                        st.success("Permanent document deleted.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")
 
    else:
        docs_list = st.session_state.temp_docs
 
        if not docs_list:
            st.info("No temp docs yet.")
            st.session_state.selected_doc_temp = None
        else:
            options = [f'{d["filename"]} | {d["doc_id"][:8]}' for d in docs_list]
 
            selected_index = 0
            if st.session_state.selected_doc_temp is not None:
                for i, d in enumerate(docs_list):
                    if d["doc_id"] == st.session_state.selected_doc_temp:
                        selected_index = i
                        break
 
            chosen = st.selectbox(
                "Choose temporary document",
                options,
                index=selected_index,
                key="temp_doc_dropdown"
            )
            st.session_state.selected_doc_temp = docs_list[options.index(chosen)]["doc_id"]
 
        c1, c2 = st.columns(2)
 
        with c1:
            if st.button("Refresh", use_container_width=True):
                st.rerun()
 
        with c2:

            
            if st.button("Delete", use_container_width=True):
                if st.session_state.selected_doc_temp is None:
                    st.warning("Select a document first.")
                else:
                    doc_id = st.session_state.selected_doc_temp
                    st.session_state.temp_stores.pop(doc_id, None)
                    st.session_state.temp_docs = [
                        d for d in st.session_state.temp_docs if d["doc_id"] != doc_id
                    ]
                    st.session_state.selected_doc_temp = None
                    st.success("Temporary document deleted.")
                    st.rerun()
 
# ---------------- Upload + Ingest ----------------
st.subheader(f"{st.session_state.mode} mode")
 
uploader_key = "perm_uploader" if st.session_state.mode == "Permanent" else "temp_uploader"
uploaded = st.file_uploader("Upload PDF", type=["pdf"], key=uploader_key)
 
if st.button("Ingest"):
    if not uploaded:
        st.error("Upload a PDF first.")
        st.stop()
 
    file_bytes = uploaded.read()
    text = bot.extract_pdf_text(file_bytes)
 
    if not text or not text.strip():
        st.error("No text extracted from PDF.")
        st.stop()
 
    filename = uploaded.name
    doc_id = str(uuid.uuid4())
 
    try:
        if st.session_state.mode == "Permanent":
            n = bot.ingest_permanent(doc_id=doc_id, filename=filename, text=text)
 
            if n == -1:
                st.warning("Document already exists.")
                st.stop()
 
            st.success(f"Ingested {n} chunks into PGVector + Azure Search")
            st.session_state.perm_docs = bot.permanent_storage.list_documents()
            st.session_state.selected_doc_perm = doc_id
 
            # clear permanent uploader after successful upload
            st.session_state.pop("perm_uploader", None)
 
            st.rerun()
 
        else:
            if any(
                d["filename"].strip().lower() == filename.strip().lower()
                for d in st.session_state.temp_docs
            ):
                st.warning("Document already exists.")
                st.stop()
 
            store, n = bot.ingest_temporary(doc_id=doc_id, filename=filename, text=text)
            st.session_state.temp_stores[doc_id] = store
            st.session_state.temp_docs.append({"doc_id": doc_id, "filename": filename})
            st.session_state.selected_doc_temp = doc_id
            st.success(f"Ingested {n} chunks into FAISS temp store")
 
            # clear temporary uploader after successful upload
            st.session_state.pop("temp_uploader", None)
 
            st.rerun()
 
    except Exception as e:
        st.error(f"Ingestion failed: {e}")
        st.stop()
 
st.divider()
 
# ---------------- Chat ----------------
st.subheader("Chat (answers only from selected document)")
 
if st.session_state.mode == "Permanent":
    chat_key = "chat_perm"
    selected_doc = st.session_state.selected_doc_perm
else:
    chat_key = "chat_temp"
    selected_doc = st.session_state.selected_doc_temp
 
for m in st.session_state[chat_key]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
 
question = st.chat_input("Ask a question about the selected document...")
 
if question:
    st.session_state[chat_key].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
 
    if not selected_doc:
        answer, sources = "Select a document first.", []
    else:
        try:
            if st.session_state.mode == "Permanent":
                answer, sources = bot.ask_from_permanent(
                    doc_id=selected_doc,
                    question=question,
                    top_k=5,
                )
            else:
                store = st.session_state.temp_stores.get(selected_doc)
                if store is None:
                    answer, sources = "Temporary document not available.", []
                else:
                    answer, sources = bot.ask_from_temporary(
                        faiss_store=store,
                        question=question,
                        top_k=5,
                    )
 
        except Exception as e:
            answer, sources = f"Chat failed: {e}", []
 
    st.session_state[chat_key].append({"role": "assistant", "content": answer})
 
    with st.chat_message("assistant"):
        st.markdown(answer)
 
    with st.expander("Sources"):
        st.write(sources)
 
    # auto delete temp after answer
    if (
        st.session_state.mode == "Temporary"
        and st.session_state.auto_delete_temp
        and selected_doc
    ):
        st.session_state.temp_stores.pop(selected_doc, None)
        st.session_state.temp_docs = [
            d for d in st.session_state.temp_docs if d["doc_id"] != selected_doc
        ]
        st.session_state.selected_doc_temp = None
        st.info("Temporary document auto-deleted.")
        st.rerun()
 