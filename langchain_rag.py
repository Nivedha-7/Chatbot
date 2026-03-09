import os
import io
from typing import List, Tuple, Dict, Optional
 
from dotenv import load_dotenv
from pypdf import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
 
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
 
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from sqlalchemy import create_engine, text as sql_text
 
load_dotenv(override=True)
 
# ---------------- Global DB Engine ----------------
PG_CONN_STR = os.environ["PG_CONN_STR"]
 
ENGINE = create_engine(
    PG_CONN_STR,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)
 
# ---------------- Azure OpenAI Service ----------------
class AzureOpenAIService:
    def __init__(self):
        self.azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
        self.api_key = os.environ["AZURE_OPENAI_API_KEY"]
        self.api_version = os.environ["AZURE_OPENAI_API_VERSION"]
        self.chat_deployment = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
        self.embed_deployment = os.environ["AZURE_OPENAI_EMBED_DEPLOYMENT"]
 
    def get_embeddings(self) -> AzureOpenAIEmbeddings:
        return AzureOpenAIEmbeddings(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
            azure_deployment=self.embed_deployment,
        )
 
    def get_llm(self) -> AzureChatOpenAI:
        return AzureChatOpenAI(
            azure_endpoint=self.azure_endpoint,
            api_key=self.api_key,
            api_version=self.api_version,
            azure_deployment=self.chat_deployment,
            temperature=0,
        )
 
# ---------------- PDF Extractor ----------------
class PDFTextExtractor:
    def __init__(self):
        # Uncomment if needed
        # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        pass
 
    def extract_text(self, file_bytes: bytes) -> str:
        # Step 1: normal PDF text extraction
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            texts = []
            for page in reader.pages:
                texts.append(page.extract_text() or "")
            text = "\n".join(texts).strip()
            if text:
                return text
        except Exception:
            pass
 
        # Step 2: OCR fallback
        try:
            images = convert_from_bytes(file_bytes)
            ocr_texts = []
            for img in images:
                ocr_texts.append(pytesseract.image_to_string(img) or "")
            return "\n".join(ocr_texts).strip()
        except Exception as e:
            raise Exception(f"OCR failed: {e}")
 
# ---------------- Chunking ----------------
class ChunkingService:
    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )
 
    def split_text(self, text: str) -> List[str]:
        return self.splitter.split_text(text)
 
# ---------------- Permanent Storage ----------------
class PermanentStorageService:
    AZS_ID_FIELD = "id"
    AZS_DOC_ID_FIELD = "doc_id"
    AZS_FILENAME_FIELD = "filename"
    AZS_CONTENT_FIELD = "content"
    AZS_VECTOR_FIELD = "contentVector"
 
    def __init__(self, openai_service: AzureOpenAIService, collection_name: str = "rag_chunks"):
        self.openai_service = openai_service
        self.collection_name = collection_name
        self.pg_conn_str = PG_CONN_STR
        self.azure_search_endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
        self.azure_search_key = os.environ["AZURE_SEARCH_KEY"]
        self.azure_search_index = os.environ["AZURE_SEARCH_INDEX"]
 
    def get_pg_vectorstore(self) -> PGVector:
        return PGVector(
            embeddings=self.openai_service.get_embeddings(),
            collection_name=self.collection_name,
            connection=self.pg_conn_str,
            use_jsonb=True,
        )
 
    def get_search_client(self) -> SearchClient:
        return SearchClient(
            endpoint=self.azure_search_endpoint,
            index_name=self.azure_search_index,
            credential=AzureKeyCredential(self.azure_search_key),
        )
 
    def list_documents(self) -> List[Dict[str, str]]:
        query = sql_text("""
            SELECT DISTINCT
                (e.cmetadata->>'doc_id') AS doc_id,
                (e.cmetadata->>'filename') AS filename
            FROM langchain_pg_embedding e
            JOIN langchain_pg_collection c ON e.collection_id = c.uuid
            WHERE c.name = :collection_name
              AND (e.cmetadata ? 'doc_id')
              AND (e.cmetadata ? 'filename')
            ORDER BY filename;
        """)
 
        with ENGINE.connect() as con:
            rows = con.execute(query, {"collection_name": self.collection_name}).fetchall()
 
        result = []
        for r in rows:
            if r.doc_id and r.filename:
                result.append({"doc_id": r.doc_id, "filename": r.filename})
        return result
 
    def document_exists(self, filename: str) -> bool:
        filename_clean = filename.strip().lower()
        docs = self.list_documents()
        return any(d["filename"].strip().lower() == filename_clean for d in docs)
 
    def store_in_pgvector(self, doc_id: str, filename: str, chunks: List[str]) -> int:
        vs = self.get_pg_vectorstore()
        docs = [
            Document(
                page_content=chunk,
                metadata={"doc_id": doc_id, "filename": filename, "chunk_id": i},
            )
            for i, chunk in enumerate(chunks)
        ]
        vs.add_documents(docs)
        return len(docs)
 
    def store_in_azure_search(self, doc_id: str, filename: str, chunks: List[str], vectors: List[List[float]]) -> int:
        client = self.get_search_client()
 
        batch = []
        for i, (chunk_text, vec) in enumerate(zip(chunks, vectors)):
            batch.append({
                self.AZS_ID_FIELD: f"{doc_id}_{i}",
                self.AZS_DOC_ID_FIELD: doc_id,
                self.AZS_FILENAME_FIELD: filename,
                self.AZS_CONTENT_FIELD: chunk_text,
                self.AZS_VECTOR_FIELD: vec,
            })
 
        results = client.upload_documents(documents=batch)
        return sum(1 for r in results if getattr(r, "succeeded", False))
 
    def delete_document(self, doc_id: str) -> bool:
        with ENGINE.begin() as con:
            con.execute(
                sql_text("""
                    DELETE FROM langchain_pg_embedding
                    WHERE cmetadata->>'doc_id' = :doc_id
                """),
                {"doc_id": doc_id}
            )
 
        client = self.get_search_client()
        results = client.search(
            search_text="*",
            filter=f"{self.AZS_DOC_ID_FIELD} eq '{doc_id}'",
            top=1000
        )
 
        docs_to_delete = []
        for r in results:
            rid = getattr(r, self.AZS_ID_FIELD, None) or r.get(self.AZS_ID_FIELD)
            if rid:
                docs_to_delete.append({self.AZS_ID_FIELD: rid})
 
        if docs_to_delete:
            client.delete_documents(documents=docs_to_delete)
 
        return True
 
    def retrieve_documents(self, doc_id: str, question: str, top_k: int = 5) -> List[Document]:
        vs = self.get_pg_vectorstore()
        retriever = vs.as_retriever(
            search_kwargs={"k": top_k, "filter": {"doc_id": doc_id}}
        )
        return retriever.invoke(question)
 
# ---------------- Temporary Storage ----------------
class TemporaryStorageService:
    def __init__(self, openai_service: AzureOpenAIService):
        self.openai_service = openai_service
 
    def build_faiss_store(self, text: str, doc_id: str, filename: str, chunker: ChunkingService) -> Tuple[FAISS, int]:
        chunks = chunker.split_text(text)
        docs = [
            Document(
                page_content=chunk,
                metadata={"doc_id": doc_id, "filename": filename, "chunk_id": i}
            )
            for i, chunk in enumerate(chunks)
        ]
        store = FAISS.from_documents(docs, self.openai_service.get_embeddings())
        return store, len(docs)
 
    def retrieve_documents(self, faiss_store: FAISS, question: str, top_k: int = 5) -> List[Document]:
        return faiss_store.similarity_search(question, k=top_k)
 
# ---------------- Main Chatbot Service ----------------
class RAGChatbotService:
    def __init__(self):
        self.openai_service = AzureOpenAIService()
        self.extractor = PDFTextExtractor()
        self.chunker = ChunkingService()
        self.permanent_storage = PermanentStorageService(self.openai_service)
        self.temporary_storage = TemporaryStorageService(self.openai_service)
 
        self.prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful assistant. Answer ONLY using the provided context. "
                "If the answer is not in the context, say exactly: Not in the document."
            ),
            ("user", "Question:\n{question}\n\nContext:\n{context}")
        ])
 
    def extract_pdf_text(self, file_bytes: bytes) -> str:
        return self.extractor.extract_text(file_bytes)
 
    def ingest_permanent(self, doc_id: str, filename: str, text: str) -> int:
        if self.permanent_storage.document_exists(filename):
            return -1
 
        chunks = self.chunker.split_text(text)
        if not chunks:
            return 0
 
        vectors = self.openai_service.get_embeddings().embed_documents(chunks)
 
        self.permanent_storage.store_in_pgvector(doc_id, filename, chunks)
        self.permanent_storage.store_in_azure_search(doc_id, filename, chunks, vectors)
 
        return len(chunks)
 
    def ingest_temporary(self, doc_id: str, filename: str, text: str) -> Tuple[FAISS, int]:
        return self.temporary_storage.build_faiss_store(text, doc_id, filename, self.chunker)
 
    def ask_from_permanent(self, doc_id: str, question: str, top_k: int = 5) -> Tuple[str, list]:
        docs = self.permanent_storage.retrieve_documents(doc_id, question, top_k)
        return self._generate_answer(question, docs)
 
    def ask_from_temporary(self, faiss_store: FAISS, question: str, top_k: int = 5) -> Tuple[str, list]:
        docs = self.temporary_storage.retrieve_documents(faiss_store, question, top_k)
        return self._generate_answer(question, docs)
 
    def _generate_answer(self, question: str, docs: List[Document]) -> Tuple[str, list]:
        if not docs:
            return "Not in the document.", []
 
        context = "\n\n".join([d.page_content for d in docs]).strip()
        if not context:
            return "Not in the document.", []
 
        llm = self.openai_service.get_llm()
        msg = self.prompt.format_messages(question=question, context=context)
        resp = llm.invoke(msg)
 
        return resp.content, [d.metadata for d in docs]
 