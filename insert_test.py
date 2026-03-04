import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("PG_HOST"),
    port=os.getenv("PG_PORT"),
    dbname=os.getenv("PG_DATABASE"),
    user=os.getenv("PG_USER"),
    password=os.getenv("PG_PASSWORD"),
    sslmode="require" 
)

cur = conn.cursor()
cur.execute("INSERT INTO documents (filename, content) VALUES (%s, %s)", ("test.pdf", "This is a test document"))

conn.commit()
print("Inserted successfully")

cur.close()
conn.close()
    