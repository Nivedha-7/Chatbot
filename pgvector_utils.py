import os
import uuid
from typing import List, Optional, Tuple
 
import psycopg2
import psycopg2.extras
 
def to_pgvector(vec):
    return "[" + ",".join(map(str,vec))+ "]"
 
# ---------- Connection ----------
 
def get_conn():
     dsn = os.getenv("PG_DSN")
     if dsn and dsn.strip():
         return psycopg2.connect(dsn)
     
     return psycopg2.connect(
        host  = os.getenv("PG_HOST"),
        port = int(os.getenv("PG_PORT","5432")),
        dbname = os.getenv("PG_DATABASE","postgres"),
        user = os.getenv("PG_USER"),
        password = os.getenv("PG_PASSWORD"),
        sslmode = os.getenv("PG_SSLMODE","require")
    )
 
def _to_pgvector_literal(vec):
    vec = list(map(float, vec))
    return "[" + ",".join(f"{float(x):.8f}" for x in vec) + "]"

def vector_seacrh_in_doc(doc_id:str, q_vec, top_k:int=5):
    qv = _to_pgvector_literal(q_vec)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, doc_id, chunk_index,content 
        FROM doc_chunks
        WHERE doc_id = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %S::int;
        """,
        (doc_id,qv,top_k)
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows
 
# ---------- DB Setup ----------
 
import os
import psycopg2
 
 
def init_db():
 
    conn = psycopg2.connect(
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT"),
        database=os.getenv("PG_DATABASE"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        sslmode=os.getenv("PG_SSLMODE")
    )
 
    cur = conn.cursor()
 
    # enable vector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
 
    # documents table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT,
            blob_url TEXT
        );
    """)
 
    # chunks table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS doc_chunks (
            id SERIAL PRIMARY KEY,
            doc_id TEXT REFERENCES documents(id),
            chunk_text TEXT,
            embedding VECTOR(1536)
        );
    """)
 
    conn.commit()
    cur.close()
    conn.close()
 
    print("Postgres tables created/verified successfully")
 
 
# ---------- Parent insert/upsert ----------
 
def upsert_document(doc_id: str, filename: str, blob_url: Optional[str] = None):
    """
    Ensures the parent row exists BEFORE inserting chunks.
    """
    conn = get_conn()
    cur = conn.cursor()
 
    cur.execute(
        """
        INSERT INTO documents (id, filename, blob_url)
        VALUES (%s, %s, %s)
        ON CONFLICT (id)
        DO UPDATE SET filename = EXCLUDED.filename,
                      blob_url = COALESCE(EXCLUDED.blob_url, documents.blob_url);
        """,
        (doc_id, filename, blob_url)
    )
 
    conn.commit()
    cur.close()
    conn.close()
 
 
# ---------- Child insert ----------
 
def insert_chunks(doc_id: str, chunks: List[str], vectors: List[List[float]], filename: Optional[str] = None):
    """
    Inserts chunks for a doc_id.
 
    ✅ IMPORTANT: If the document row is missing, we create it here too (failsafe),
    so ForeignKeyViolation will NEVER happen again.
    """
    if len(chunks) != len(vectors):
        raise ValueError(f"chunks and vectors length mismatch: {len(chunks)} vs {len(vectors)}")
 
    conn = get_conn()
    cur = conn.cursor()
 
    # FAILSAFE: make sure parent exists
    cur.execute(
        """
        INSERT INTO documents (id, filename)
        VALUES (%s, %s)
        ON CONFLICT (id) DO NOTHING;
        """,
        (doc_id, filename or "unknown")
    )
 
    rows = []
    for i, (chunk_text, vec) in enumerate(zip(chunks, vectors)):
        chunk_id = f"{doc_id}_{i}"  
        rows.append((chunk_id,doc_id,i,chunk_text,vec))
    
 
    psycopg2.extras.execute_values(
        cur,
        """
        INSERT INTO doc_chunks (id, doc_id, chunk_index, content, embedding)
        VALUES %s
        ON CONFLICT (id)
        DO UPDATE SET
            content = EXCLUDED.content,
            embedding = EXCLUDED.embedding;
        """,
        rows,
        page_size=200
    )
 
    conn.commit()
    cur.close()
    conn.close()
 
 
# ---------- Utilities ----------


def list_documents():
    import psycopg2, os
    conn = psycopg2.connect(os.getenv("PG_DSN"))
    cur = conn.cursor()
    cur.execute("SELECT id, filename, created_at FROM documents ORDER BY created_at DESC")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"id": r[0], "filename": r[1], "created_at": r[2]} for r in rows]
 
 
def get_chunks_for_doc(doc_id: str, limit: int = 5):
    import psycopg2, os
    conn = psycopg2.connect(os.getenv("PG_DSN"))
    cur = conn.cursor()
    cur.execute(
        "SELECT content FROM doc_chunks WHERE doc_id=%s ORDER BY chunk_index ASC LIMIT %s",
        (doc_id, limit),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [r[0] for r in rows]
 
 
def vector_search_in_doc(doc_id: str, query_vec, top_k: int = 5):
    import psycopg2, os
    conn = psycopg2.connect(os.getenv("PG_DSN"))
    cur = conn.cursor()
 
    # pgvector distance operator: <-> (L2) or <=> (cosine) depending on your setup
    # Using cosine distance:
    cur.execute(
        """
        SELECT content, chunk_index
        FROM doc_chunks
        WHERE doc_id = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (doc_id, query_vec, top_k),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
 
    return [{"content": r[0], "chunk_index": r[1]} for r in rows]