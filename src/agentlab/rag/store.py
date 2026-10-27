"""pgvector: схема, вставка, векторный поиск."""
import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

from src import settings
from ..llm import embed


def connect(register: bool = True):
    conn = psycopg2.connect(**settings.PG)
    if register:
        register_vector(conn)  # тип vector должен уже существовать в базе
    return conn


def setup_database() -> None:
    conn = connect(register=False)  # расширения ещё может не быть
    with conn, conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                doc_id TEXT NOT NULL,
                title TEXT NOT NULL,
                chunk_index INT NOT NULL,
                text TEXT NOT NULL,
                embedding vector({settings.EMBEDDING_DIM}) NOT NULL,
                UNIQUE (doc_id, chunk_index)
            );
        """)
        # HNSW, а не ivfflat: работает на пустой таблице и не требует обучения индекса
        cur.execute("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
            ON chunks USING hnsw (embedding vector_cosine_ops);
        """)
    conn.close()


def reset_chunks() -> None:
    conn = connect()
    with conn, conn.cursor() as cur:
        cur.execute("TRUNCATE chunks RESTART IDENTITY;")
    conn.close()


def insert_chunks(chunks: list[dict]) -> None:
    vectors = embed([c["text"] for c in chunks])
    rows = [
        (c["doc_id"], c["title"], c["chunk_index"], c["text"], np.array(v, dtype=np.float32))
        for c, v in zip(chunks, vectors)
    ]
    conn = connect()
    with conn, conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO chunks (doc_id, title, chunk_index, text, embedding) VALUES %s",
            rows,
        )
    conn.close()
    print(f"Сохранено {len(rows)} chunks")


def fetch_all_chunks() -> list[dict]:
    conn = connect()
    with conn.cursor() as cur:
        cur.execute("SELECT doc_id, title, chunk_index, text FROM chunks ORDER BY id;")
        rows = cur.fetchall()
    conn.close()
    return [
        {"doc_id": r[0], "title": r[1], "chunk_index": r[2], "text": r[3]} for r in rows
    ]


def vector_search(query: str, top_k: int = 5) -> list[dict]:
    q = np.array(embed([query])[0], dtype=np.float32)
    conn = connect()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT doc_id, title, chunk_index, text, embedding <=> %s AS distance
            FROM chunks
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (q, q, top_k),
        )
        rows = cur.fetchall()
    conn.close()
    return [
        {"doc_id": r[0], "title": r[1], "chunk_index": r[2], "text": r[3], "distance": float(r[4])}
        for r in rows
    ]