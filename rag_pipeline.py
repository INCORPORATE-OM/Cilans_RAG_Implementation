import os
import functools
from google import genai
from google.genai import types
from chunking_strategy import MultilingualEmbedder
from database_setup import get_chroma_client

# ─── In-process LRU cache: exact same query → skip API call entirely ───
@functools.lru_cache(maxsize=128)
def _cached_generate(query: str, context_hash: str, context: str,
                     language_hint: str, model_name: str, api_key: str) -> str:
    """Inner function — cached by (query, context, lang). Only hits API on new combos."""
    client = genai.Client(api_key=api_key)

    lang_instruction = {
        "en": "Respond in English.",
        "gu": "Respond in Gujarati (ગુજરાતી).",
        "hi": "Respond in Hindi."
    }.get(language_hint, "Respond in English.")

    # Compact system prompt — fewer tokens without losing guidance quality
    system = (
        "You are a guide to the Yatharth Geeta (Gujarati/Sanskrit commentary on Bhagavad Gita). "
        "Answer ONLY from the context below. Quote Shloka [num] when relevant. "
        "If context is insufficient, say so. Do not speculate."
    )

    full_prompt = (
        f"{system}\n\n"
        f"=== CONTEXT ===\n{context}\n\n"
        f"=== QUESTION ===\n{query}\n\n"
        f"{lang_instruction}"
    )

    response = client.models.generate_content(
        model=model_name,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=500,  # Reduced from 800 → saves tokens
        )
    )
    return response.text


def retrieve_chunks(
    query: str,
    embedder: MultilingualEmbedder,
    top_k: int = 3,
    chapter_filter: int = None,
    script_filter: str = None,
    similarity_threshold: float = 0.3
) -> list[dict]:
    """Retrieves most relevant chunks using ChromaDB Local Vector Search."""
    client = get_chroma_client()
    collection = client.get_collection(name="geeta_chunks")

    query_embedding = embedder.embed_query(query)

    where_clause = {}
    if chapter_filter and chapter_filter != "All":
        where_clause["chapter"] = chapter_filter
    if script_filter and script_filter != "All":
        where_clause["script"] = script_filter

    where = where_clause if where_clause else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    vector_results = []
    if results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            dist = results["distances"][0][i]
            sim = max(0.0, 1.0 - dist)
            meta = results["metadatas"][0][i]
            # Truncate each chunk to 600 chars max before sending to LLM
            raw_text = results["documents"][0][i]
            trimmed_text = raw_text[:600] + ("…" if len(raw_text) > 600 else "")
            vector_results.append({
                "chunk_id": results["ids"][0][i],
                "page_num": meta.get("page_num", 0),
                "verse_num": None if meta.get("verse_num", -1) == -1 else meta.get("verse_num"),
                "chunk_type": meta.get("chunk_type", ""),
                "script": meta.get("script", ""),
                "chapter": meta.get("chapter", 0),
                "text": trimmed_text,
                "full_text": raw_text,      # keep original for display
                "similarity": sim
            })

    return vector_results


def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    language_hint: str = "en"
) -> dict:
    """Token-efficient RAG answer generation with in-process result caching."""
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set. Please add it to your .env file.")

    model_name = os.getenv("LLM_MODEL", "gemini-2.0-flash")

    # Build compact context — only trimmed text is sent to the model
    context_parts = []
    for chunk in retrieved_chunks:
        label = f"[Pg{chunk['page_num']}"
        if chunk["verse_num"]:
            label += f"/Shloka{chunk['verse_num']}"
        label += f" sim={chunk['similarity']:.2f}]"
        context_parts.append(f"{label} {chunk['text']}")

    context = "\n\n".join(context_parts)
    context_hash = str(hash(context))  # lru_cache key for context

    answer_text = _cached_generate(
        query=query,
        context_hash=context_hash,
        context=context,
        language_hint=language_hint,
        model_name=model_name,
        api_key=api_key,
    )

    return {
        "query": query,
        "answer": answer_text,
        "sources": [
            {
                "chunk_id": c["chunk_id"],
                "page": c["page_num"],
                "verse": c["verse_num"],
                "similarity": round(c["similarity"], 3)
            }
            for c in retrieved_chunks
        ],
        "model": model_name
    }
