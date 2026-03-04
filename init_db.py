import psycopg2
from psycopg2 import sql

# SQL statements to create tables
CREATE_DOCUMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename TEXT,
    content TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_EMBEDDINGS_TABLE = """
CREATE TABLE IF NOT EXISTS embeddings (
    id SERIAL PRIMARY KEY,
    document_id INT,
    embedding VECTOR(1536)
);
"""

def init_db(conn):
    """Initialize the database tables"""
    cur = conn.cursor()
    try:
        cur.execute(CREATE_DOCUMENTS_TABLE)
        cur.execute(CREATE_EMBEDDINGS_TABLE)
        conn.commit()
        print("Tables created successfully")
    except Exception as e:
        print(f"Error creating tables: {e}")
        conn.rollback()
    finally:
        cur.close()   