from dotenv import load_dotenv
load_dotenv()  # MUST be first — before any genai/embedding imports

import streamlit as st
from chunking_strategy import MultilingualEmbedder
from rag_pipeline import retrieve_chunks, generate_answer

st.set_page_config(page_title="Yatharth Geeta RAG", page_icon="🕉️")
st.title("🕉️ Yatharth Geeta — Knowledge Explorer")

# ─── CRITICAL: Cache the embedding model so it only loads ONCE per session ───
@st.cache_resource(show_spinner="Loading embedding model (one-time)...")
def get_embedder():
    return MultilingualEmbedder()

# ─── Cache LLM answers so identical queries don't burn API quota ───
@st.cache_data(show_spinner=False, ttl=3600)  # cache answers for 1 hour
def cached_answer(query: str, top_k: int, chapter, script, lang: str):
    embedder = get_embedder()
    chunks = retrieve_chunks(
        query, embedder,
        top_k=top_k,
        chapter_filter=chapter,
        script_filter=script
    )
    result = generate_answer(query, chunks, language_hint=lang)
    return result, chunks

# ─── Sidebar ───
with st.sidebar:
    st.header("Filters")
    chapter = st.selectbox(
        "Chapter", [None, 1, 2, 3],
        format_func=lambda x: "All" if x is None else f"Chapter {x}"
    )
    script = st.selectbox(
        "Script", [None, "gujarati", "devanagari", "latin"],
        format_func=lambda x: "All" if x is None else x.capitalize()
    )
    lang = st.selectbox(
        "Response Language", ["en", "gu", "hi"],
        format_func=lambda x: {"en": "English", "gu": "Gujarati", "hi": "Hindi"}[x]
    )
    # Default to 3 passages — keeps tokens low, answers still high quality
    top_k = st.slider("Retrieved passages", 1, 6, 3)

    st.divider()
    st.caption("📌 Lower passages = fewer API tokens used")

# ─── Main input ───
query = st.text_input(
    "Ask a question about the Geeta:",
    placeholder="What is the meaning of Dharmashetra?"
)

if query:
    with st.spinner("Searching..."):
        result, chunks = cached_answer(query, top_k, chapter, script, lang)

    st.markdown("### Answer")
    st.write(result["answer"])
    st.caption(f"Model: `{result['model']}` · Sources: {len(result['sources'])}")

    with st.expander("Source passages"):
        for src in result["sources"]:
            st.markdown(
                f"**Page {src['page']}** | Verse {src['verse']} | Similarity: `{src['similarity']}`"
            )
            matching = next((c["text"] for c in chunks if c["chunk_id"] == src["chunk_id"]), "")
            st.text(matching[:300] + "...")
