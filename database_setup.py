import os
import chromadb
from chunking_strategy import Chunk, MultilingualEmbedder

def get_chroma_client():
    # Persist the vector database locally in a folder named 'chroma_db'
    return chromadb.PersistentClient(path="./chroma_db")

def create_schema():
    """Creates the vector store table and index."""
    client = get_chroma_client()
    # Cosine similarity is generally preferred for embedding models
    client.get_or_create_collection(
        name="geeta_chunks",
        metadata={"hnsw:space": "cosine"}
    )
    print("Local ChromaDB schema created successfully in ./chroma_db")

def ingest_chunks(chunks: list[Chunk], embedder: MultilingualEmbedder):
    """
    Embeds all chunks and inserts into ChromaDB.
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(name="geeta_chunks")
    
    # Extract texts and embed in batch
    texts = [c.text for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = embedder.embed_batch(texts, batch_size=16)
    
    ids = []
    metadatas = []
    
    for c in chunks:
        ids.append(c.chunk_id)
        # Chroma metadata strictly requires str, int, float, or bool values
        meta = {
            "page_num": c.page_num,
            "verse_num": c.verse_num if c.verse_num is not None else -1,
            "chunk_type": c.chunk_type,
            "script": c.script,
            "token_count": c.token_count,
            "chapter": c.metadata.get("chapter", 0),
            "source": c.metadata.get("source", "unknown")
        }
        metadatas.append(meta)
    
    print("Saving to local ChromaDB...")
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=texts
    )
    print(f"Ingested {len(ids)} chunks into local ChromaDB.")
