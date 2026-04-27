from dataclasses import dataclass
from typing import Optional
from document_processing import detect_structure
import tiktoken
from sentence_transformers import SentenceTransformer

@dataclass
class Chunk:
    chunk_id: str
    page_num: int
    verse_num: Optional[int]
    chunk_type: str          # 'shloka', 'commentary', 'shloka+commentary'
    script: str
    text: str
    token_count: int
    metadata: dict

def infer_chapter(page: dict) -> int:
    """Rough chapter detection from page markers — customize per document."""
    # Geeta has 18 chapters; for demo, approximate by page range
    page_num = page.get('page_num', 1)
    if page_num <= 12:
        return 1
    return (page_num // 15) + 1

def split_with_overlap(text: str, enc, max_tokens: int, overlap: int) -> list[str]:
    """Splits text into token-bounded chunks with sentence overlap."""
    paragraphs = text.split('\n\n')
    result = []
    current = []
    current_tokens = 0
    
    for para in paragraphs:
        para_tokens = len(enc.encode(para))
        if current_tokens + para_tokens > max_tokens and current:
            result.append('\n\n'.join(current))
            # Keep last paragraph for overlap
            current = current[-1:] if overlap > 0 else []
            current_tokens = len(enc.encode('\n\n'.join(current)))
        current.append(para)
        current_tokens += para_tokens
    
    if current:
        result.append('\n\n'.join(current))
    return result

def create_chunks(pages: list[dict], max_tokens: int = 400) -> list[Chunk]:
    """
    Strategy:
    1. Primary: pair each shloka with its following commentary (semantic unit)
    2. If commentary too long: split at natural paragraph breaks
    3. If shloka-only page: chunk by paragraphs with 50-token overlap
    """
    enc = tiktoken.get_encoding("cl100k_base")
    
    chunks = []
    chunk_idx = 0
    
    for page in pages:
        segments = detect_structure(page['text'])
        
        i = 0
        while i < len(segments):
            seg = segments[i]
            
            if seg['type'] == 'shloka' and i + 1 < len(segments):
                # Pair shloka with next commentary
                next_seg = segments[i + 1]
                combined = f"[Shloka {seg.get('verse_num', '')}]\n{seg['content']}\n\n[Commentary]\n{next_seg['content']}"
                tokens = len(enc.encode(combined))
                
                if tokens <= max_tokens:
                    chunks.append(Chunk(
                        chunk_id=f"p{page['page_num']}_c{chunk_idx}",
                        page_num=page['page_num'],
                        verse_num=seg.get('verse_num'),
                        chunk_type='shloka+commentary',
                        script=page['script'],
                        text=combined,
                        token_count=tokens,
                        metadata={"chapter": infer_chapter(page), "source": page.get("source", "unknown")}
                    ))
                    chunk_idx += 1
                    i += 2
                    continue
            
            # Standalone segment — paragraph split if needed
            text = seg['content']
            tokens = len(enc.encode(text))
            if tokens <= max_tokens:
                chunks.append(Chunk(
                    chunk_id=f"p{page['page_num']}_c{chunk_idx}",
                    page_num=page['page_num'],
                    verse_num=seg.get('verse_num'),
                    chunk_type=seg['type'],
                    script=page['script'],
                    text=text,
                    token_count=tokens,
                    metadata={"chapter": infer_chapter(page), "source": page.get("source", "unknown")}
                ))
                chunk_idx += 1
            else:
                # Split long commentary by paragraphs with overlap
                sub_chunks = split_with_overlap(text, enc, max_tokens, overlap=50)
                for j, sub in enumerate(sub_chunks):
                    chunks.append(Chunk(
                        chunk_id=f"p{page['page_num']}_c{chunk_idx}",
                        page_num=page['page_num'],
                        verse_num=seg.get('verse_num'),
                        chunk_type=f"{seg['type']}_part{j+1}",
                        script=page['script'],
                        text=sub,
                        token_count=len(enc.encode(sub)),
                        metadata={"chapter": infer_chapter(page), "source": page.get("source", "unknown")}
                    ))
                    chunk_idx += 1
            i += 1
    
    return chunks

class MultilingualEmbedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.dimension}")
    
    def embed(self, text: str) -> list[float]:
        """
        multilingual-e5 requires 'query: ' prefix for queries,
        'passage: ' prefix for documents being indexed.
        """
        prefixed = f"passage: {text}"
        embedding = self.model.encode(prefixed, normalize_embeddings=True)
        return embedding.tolist()
    
    def embed_query(self, query: str) -> list[float]:
        prefixed = f"query: {query}"
        embedding = self.model.encode(prefixed, normalize_embeddings=True)
        return embedding.tolist()
    
    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        embeddings = self.model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True
        )
        return embeddings.tolist()
