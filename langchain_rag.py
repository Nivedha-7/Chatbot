# langchain_rag.py
import os
from typing import List, Tuple
 
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
 
# FAISS (temporary)
from langchain_community.vectorstores import FAISS
 
load_dotenv(override=True)
 
# ---------------- Azure OpenAI ----------------
def get_embeddings() -> AzureOpenAIEmbeddings:
    return AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["AZURE_OPENAI_EMBED_DEPLOYMENT"],  # deployment name
    )
 
def get_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],  # deployment name
        temperature=0,
    )
 
# ---------------- PGVector (permanent) ----------------
def get_pg_vectorstore(collection_name: str = "rag_chunks") -> PGVector:
    conn = os.environ["PG_CONN_STR"]  # SQLAlchemy-style conn string
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=collection_name,
        connection=conn,
        use_jsonb=True,
    )
 
# ---------------- Text split ----------------
def split_text(text: str, chunk_size: int = 900, chunk_overlap: int = 150) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_text(text)
 
# ---------------- Prompt ----------------
PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer ONLY using the provided context. "
     "If the answer is not in the context, say exactly: Not in the document."),
    ("user", "Question:\n{question}\n\nContext:\n{context}")
])
 
# ---------------- Ingest (PGVector) ----------------
def ingest_text_pg(doc_id: str, filename: str, text: str, collection_name: str = "rag_chunks") -> int:
    vs = get_pg_vectorstore(collection_name)
    chunks = split_text(text)
 
    docs = [
        Document(
            page_content=chunk,
            metadata={"doc_id": doc_id, "filename": filename, "chunk_id": i}
        )
        for i, chunk in enumerate(chunks)
    ]
 
    vs.add_documents(docs)
    return len(docs)
 
# ---------------- Ask (PGVector) ----------------
def ask_question_pg(
    doc_id: str,
    question: str,
    top_k: int = 5,
    collection_name: str = "rag_chunks"
) -> Tuple[str, list]:
    vs = get_pg_vectorstore(collection_name)
 
    # ✅ Filter so we search ONLY in the selected document
    # Works reliably with langchain_postgres PGVector
    docs = vs.similarity_search(
        query=question,
        k=top_k,
        filter={"doc_id": doc_id}
    )
 
    if not docs:
        return "Not in the document.", []
 
    context = "\n\n".join([d.page_content for d in docs]).strip()
    if not context:
        return "Not in the document.", []
 
    llm = get_llm()
    msg = PROMPT.format_messages(question=question, context=context)
    resp = llm.invoke(msg)
 
    return resp.content, [d.metadata for d in docs]
 
# ===================== TEMP (FAISS) =====================
 
def build_faiss_for_doc(text: str, doc_id: str, filename: str) -> Tuple[FAISS, int]:
    chunks = split_text(text)
    docs = [
        Document(page_content=chunk, metadata={"doc_id": doc_id, "filename": filename, "chunk_id": i})
        for i, chunk in enumerate(chunks)
    ]
    faiss_store = FAISS.from_documents(docs, get_embeddings())
    return faiss_store, len(docs)
 
def ask_question_faiss(faiss_store: FAISS, question: str, top_k: int = 5) -> Tuple[str, list]:
    docs = faiss_store.similarity_search(question, k=top_k)
 
    if not docs:
        return "Not in the document.", []
 
    context = "\n\n".join([d.page_content for d in docs]).strip()
    if not context:
        return "Not in the document.", []
 
    llm = get_llm()
    msg = PROMPT.format_messages(question=question, context=context)
    resp = llm.invoke(msg)
 
    return resp.content, [d.metadata for d in docs]
 