import os
from dotenv import load_dotenv
from document_processing import extract_with_ocr_fallback, preprocess_pages, export_to_markdown
from chunking_strategy import create_chunks, MultilingualEmbedder
from database_setup import create_schema, ingest_chunks
from rag_pipeline import retrieve_chunks, generate_answer

load_dotenv()

def run_ingestion_pipeline(pdf_path: str):
    """Runs full ingestion: PDF → OCR → Chunks → Embeddings → pgvector."""
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    print("=== Step 1: OCR + Preprocessing ===")
    raw_pages = extract_with_ocr_fallback(pdf_path)
    clean_pages = preprocess_pages(raw_pages)
    
    os.makedirs("output", exist_ok=True)
    export_to_markdown(clean_pages, "output/clean_text.md")
    
    print(f"\n=== Step 2: Chunking ({len(clean_pages)} pages) ===")
    chunks = create_chunks(clean_pages, max_tokens=400)
    print(f"Created {len(chunks)} chunks")
    
    print("\n=== Step 3: Setting up pgvector ===")
    create_schema()
    
    print("\n=== Step 4: Embedding + Ingestion ===")
    embedder = MultilingualEmbedder()
    ingest_chunks(chunks, embedder)
    
    print("\n Ingestion complete. Vector store ready.")

def run_qa(query: str, **kwargs):
    """Single query → retrieve → answer."""
    embedder = MultilingualEmbedder()
    chunks = retrieve_chunks(query, embedder, top_k=5, **kwargs)
    result = generate_answer(query, chunks)
    
    print(f"\nQ: {result['query']}")
    print(f"\nA: {result['answer']}")
    print(f"\nSources: {result['sources']}")
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python main.py [ingest|query] [args]")
        sys.exit(1)
        
    command = sys.argv[1]
    
    if command == "ingest":
        if len(sys.argv) < 3:
            print("Usage: python main.py ingest <path/to/pdf>")
            sys.exit(1)
        run_ingestion_pipeline(sys.argv[2])
    elif command == "query":
        if len(sys.argv) < 3:
            print("Usage: python main.py query <your query string>")
            sys.exit(1)
        run_qa(" ".join(sys.argv[2:]))
