# pdf_esg_inference.py
import fitz  # PyMuPDF
from infer_text import infer_text

# Example keyword set for pre-filtering
ESG_KEYWORDS = {
    "oil spill", "product recall", "data breach", "strike", "pollution",
    "workplace safety", "environmental fine", "fraud", "corruption"
}

def extract_text_chunks(pdf_path, chunk_size=500):
    """Yield text chunks from PDF pages."""
    doc = fitz.open(pdf_path)
    for page in doc:
        text = page.get_text()
        # Split into chunks
        for i in range(0, len(text), chunk_size):
            yield text[i:i+chunk_size]

def prefilter_chunk(chunk):
    """Return True if chunk contains any ESG keyword."""
    lower_chunk = chunk.lower()
    return any(kw in lower_chunk for kw in ESG_KEYWORDS)

def analyze_pdf(pdf_path):
    results = []
    for chunk in extract_text_chunks(pdf_path):
        if prefilter_chunk(chunk):
            probs = infer_text(chunk)
            results.append({"text": chunk, "probs": probs})
    return results

# Example usage
if __name__ == "__main__":
    pdf_file = "data/Shell_ESG_Report_2025.pdf"
    events = analyze_pdf(pdf_file)
    for e in events[:3]:  # print first 3
        print(e)