CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
	filename TEXT,
	content TEXT,
	created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
	document_id INT,
	embedding VECTOR(1536)
);