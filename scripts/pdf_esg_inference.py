import fitz  
from infer_text import infer_text

def extract_text_chunks(pdf_path, chunk_size=500):
    """Yield text chunks from PDF pages."""
    doc = fitz.open(pdf_path)
    for page in doc:
        text = page.get_text()
        # Split into chunks
        for i in range(0, len(text), chunk_size):
            yield text[i:i+chunk_size]

def analyze_pdf(pdf_path):
    results = []
    for chunk in extract_text_chunks(pdf_path):
        probs = infer_text(chunk)
        max_cat = max(probs, key=probs.get)
        results.append({
            "text": chunk.strip(),
            "predicted_category": max_cat,
            "probs": probs
        })
    return results

def pretty_print(events, top_n=3):
    for i, e in enumerate(events[:top_n]):
        print(f"\n--- Event {i+1} ---")
        print(f"Assigned Category: {e['predicted_category']}")
        print("Top Probabilities:")
        sorted_probs = sorted(e['probs'].items(), key=lambda x: x[1], reverse=True)[:3]
        for cat, p in sorted_probs:
            print(f"  {cat}: {p:.3f}")
        print("\nExcerpt from report:")
        print(f"{e['text'][:300]}...")  

if __name__ == "__main__":
    pdf_file = "data/Shell_ESG_Report_2025.pdf"
    events = analyze_pdf(pdf_file)
    pretty_print(events, top_n=5)
