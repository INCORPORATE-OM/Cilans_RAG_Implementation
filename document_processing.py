import re
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from langdetect import detect
import os
import shutil

# Try to auto-configure tesseract path for common Windows locations if not in PATH
if not shutil.which("tesseract"):
    default_tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(default_tesseract_cmd):
        pytesseract.pytesseract.tesseract_cmd = default_tesseract_cmd

# Unicode ranges for script detection
GUJARATI_RANGE = (0x0A80, 0x0AFF)
DEVANAGARI_RANGE = (0x0900, 0x097F)

def extract_embedded_text(pdf_path: str) -> list[dict]:
    """
    Extracts text page-by-page. Returns list of {page_num, text, has_content}.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        pages.append({
            "page_num": i + 1,
            "text": text,
            "has_content": len(text) > 50  # flag sparse pages for OCR fallback
        })
    return pages

def ocr_page(image: Image.Image) -> str:
    """
    Multi-language OCR: tries Gujarati + Sanskrit + English together.
    Tesseract language codes: guj (Gujarati), san (Sanskrit), eng (English)
    """
    custom_config = r'--oem 3 --psm 6'   # OEM 3 = LSTM, PSM 6 = uniform block
    try:
        text = pytesseract.image_to_string(
            image,
            lang='guj+san+eng',
            config=custom_config
        )
        return text.strip()
    except pytesseract.pytesseract.TesseractNotFoundError:
        print("\n[!] Error: Tesseract OCR is missing or not in PATH! Skipping OCR.")
        print("[!] Install Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("[!] Ensure you select Gujarati and Sanskrit under 'Additional Languages' during install.\n")
        return ""

def extract_with_ocr_fallback(pdf_path: str) -> list[dict]:
    embedded = extract_embedded_text(pdf_path)
    doc = fitz.open(pdf_path)
    
    results = []
    for i, page_data in enumerate(embedded):
        if page_data["has_content"]:
            page_data["source"] = "embedded"
            results.append(page_data)
        else:
            # Fallback: Instead of pdf2image/poppler, use PyMuPDF to render the image
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=300) # High DPI for accurate OCR
            
            # Use 'RGB' mode directly if pix is RGB, otherwise support alpha
            mode = "RGBA" if pix.alpha else "RGB"
            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            
            ocr_text = ocr_page(img)
            results.append({
                "page_num": page_data["page_num"],
                "text": ocr_text,
                "has_content": len(ocr_text) > 20,
                "source": "ocr"
            })
    return results

def detect_script(text: str) -> str:
    """Returns 'gujarati', 'devanagari', 'latin', or 'mixed'."""
    gu = sum(1 for c in text if GUJARATI_RANGE[0] <= ord(c) <= GUJARATI_RANGE[1])
    dev = sum(1 for c in text if DEVANAGARI_RANGE[0] <= ord(c) <= DEVANAGARI_RANGE[1])
    total = max(len(text), 1)
    if gu / total > 0.3:
        return 'gujarati'
    if dev / total > 0.3:
        return 'devanagari'
    return 'latin'

def clean_text(raw: str) -> str:
    """
    Cleans OCR artifacts while preserving Unicode multilingual characters.
    """
    text = re.sub(r'^\d+\s+[\w\s]+$', '', raw, flags=re.MULTILINE)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^\s*[^\w\u0A80-\u0AFF\u0900-\u097F]{1,5}\s*$', '', text, flags=re.MULTILINE)
    return text.strip()

def preprocess_pages(pages: list[dict]) -> list[dict]:
    """Cleans and annotates each page with script metadata."""
    processed = []
    for p in pages:
        cleaned = clean_text(p["text"])
        script = detect_script(cleaned)
        processed.append({
            **p,
            "text": cleaned,
            "script": script,
            "char_count": len(cleaned)
        })
    return processed

def detect_structure(text: str) -> list[dict]:
    """
    Identifies structural elements: shlokas, commentary, verse numbers.
    Returns list of {type, content, verse_num}.
    """
    segments = []
    shloka_pattern = re.compile(r'(.*?JJ\d*JJ)', re.DOTALL)
    verse_num_pattern = re.compile(r'JJ(\d+)JJ')
    
    lines = text.split('\n')
    current_block = []
    current_type = 'commentary'
    
    for line in lines:
        if verse_num_pattern.search(line):
            if current_block:
                segments.append({
                    "type": current_type,
                    "content": '\n'.join(current_block).strip()
                })
            verse_match = verse_num_pattern.search(line)
            segments.append({
                "type": "shloka",
                "content": line.strip(),
                "verse_num": int(verse_match.group(1)) if verse_match else None
            })
            current_block = []
            current_type = 'commentary'
        else:
            current_block.append(line)
    
    if current_block:
        segments.append({"type": current_type, "content": '\n'.join(current_block).strip()})
    
    return [s for s in segments if s["content"]]

def export_to_markdown(pages: list[dict], output_path: str):
    """Writes structured clean text as hierarchical markdown."""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for page in pages:
            f.write(f"\n\n---\n## Page {page['page_num']} [{page['script']}]\n\n")
            segments = detect_structure(page['text'])
            for seg in segments:
                if seg['type'] == 'shloka':
                    verse = seg.get('verse_num', '')
                    f.write(f"\n### Shloka {verse}\n\n> {seg['content']}\n\n")
                else:
                    f.write(f"{seg['content']}\n\n")
    print(f"Clean markdown saved to: {output_path}")
