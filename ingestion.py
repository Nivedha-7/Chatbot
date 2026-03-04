import io
import os
from typing import List
 
from pypdf import PdfReader
from docx import Document as DocxDocument
 
# Azure Document Intelligence OCR
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
 
 
def _ocr_pdf_with_document_intelligence(data: bytes) -> str:
    """
    OCR for scanned/image PDFs using Azure Document Intelligence (prebuilt-read).
    """
    endpoint = os.getenv("AZURE_DI_ENDPOINT")
    key = os.getenv("AZURE_DI_KEY")
 
    if not endpoint or not key:
        raise RuntimeError(
            "Missing AZURE_DI_ENDPOINT / AZURE_DI_KEY in .env. "
            "Add them and restart the terminal/VS Code."
        )
 
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
 
    poller = client.begin_analyze_document(
        model_id="prebuilt-read",
        body=data,
        content_type="application/pdf",
    )
    result = poller.result()
 
    # Collect all lines into one text
    parts = []
    if result.pages:
        for page in result.pages:
            if page.lines:
                parts.extend([line.content for line in page.lines])
 
    return "\n".join(parts).strip()
 
 
def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower().strip()
 
    # -------- PDF --------
    if name.endswith(".pdf"):
        # 1) Try normal PDF text extraction first (fast)
        try:
            reader = PdfReader(io.BytesIO(data))
            parts = []
            for page in reader.pages:
                parts.append(page.extract_text() or "")
            text = "\n".join(parts).strip()
        except Exception:
            text = ""
 
        # 2) If empty or too small, it is likely scanned -> OCR fallback
        if len(text) < 50:
            try:
                text = _ocr_pdf_with_document_intelligence(data)
            except Exception as e:
                # keep message short but useful
                raise RuntimeError(f"OCR failed using Document Intelligence: {e}")
 
        return text
 
    # -------- DOCX --------
    if name.endswith(".docx"):
        doc = DocxDocument(io.BytesIO(data))
        return "\n".join([p.text for p in doc.paragraphs]).strip()
 
    # -------- TXT fallback --------
    return data.decode("utf-8", errors="ignore").strip()
 
 
def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []
 
    chunks = []
    i = 0
    step = max(1, chunk_size - overlap)
 
    while i < len(text):
        chunk = text[i : i + chunk_size]
        chunks.append(chunk)
        i += step
 
    return chunks