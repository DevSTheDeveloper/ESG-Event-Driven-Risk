import fitz  # PyMuPDF
from infer_text import infer_text
import re

KEYWORDS = ["oil spill", "environmental fine", "pollution", "safety recall"]

def extract_chunks(pdf_path, chunk_size=500):
    doc = fitz.open(pdf_path)
    chunks = []
    for page in doc:
        text = page.get_text()
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append(chunk)
    return chunks

def keyword_prefilter(chunk):
    chunk_lower = chunk.lower()
    return any(k.lower() in chunk_lower for k in KEYWORDS)

def process_pdf(pdf_path):
    chunks = extract_chunks(pdf_path)
    results = []
    for chunk in chunks:
        if keyword_prefilter(chunk):
            res = infer_text(chunk)
            results.append({"chunk": chunk, "result": res})
    return results

if __name__ == "__main__":
    pdf_file = "sample_shell_report.pdf"
    classified_chunks = process_pdf(pdf_file)
    for c in classified_chunks:
        print(c["result"])