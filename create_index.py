import os
import psycopg2
from dotenv import load_dotenv
 
load_dotenv()
 
def main():
    conn = psycopg2.connect(
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT", "5432"),
        dbname=os.getenv("PG_DATABASE", "postgres"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASSWORD"),
        sslmode=os.getenv("PG_SSLMODE", "require"),
    )
    cur = conn.cursor()
 
    # pgvector
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
 
    # IMPORTANT: documents table must have doc_id
    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        doc_id TEXT PRIMARY KEY,
        filename TEXT,
        uploaded_at TIMESTAMP DEFAULT NOW()
    );
    """)
 
    # chunks table depends on documents(doc_id)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS doc_chunks (
        chunk_id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL REFERENCES documents(doc_id),
        chunk_text TEXT,
        embedding VECTOR(1536),
        chunk_index INT,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """)
 
    # optional index for fast vector search (IVFFLAT needs ANALYZE after data insert)
    cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_doc_chunks_embedding
    ON doc_chunks USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    """)
 
    conn.commit()
    cur.close()
    conn.close()
    print("✅ pgvector enabled + tables created")
 
if __name__ == "__main__":
    main()
 