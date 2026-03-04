import os
import psycopg2
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).with_name(".env"),override=True)
print("PG_USER =",os.getenv("PG_USER"))
print("PG_HOST =",os.getenv("PG_HOST"))
print("PG_PORT =",os.getenv("PG_PORT"))
print("PG_PASSWORD length =")
print(len(os.getenv("PG_PASSWORD") or""))

conn = psycopg2.connect(
    host=os.getenv("PG_HOST"),
    port=int(os.getenv("PG_PORT",'5432')),
    database=os.getenv("PG_DATABASE"),
    user=os.getenv("PG_USER"),
    password=os.getenv("PG_PASSWORD"),
    sslmode=os.getenv("PG_SSLMODE","require")
)

cur = conn.cursor()
cur.execute("SELECT now();")
print("Connected! Server time:", cur.fetchone()[0])


cur.close()
conn.close()