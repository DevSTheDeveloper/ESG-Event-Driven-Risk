# scripts/infer_text.py
from load_model import get_model
import torch

tokenizer, model, device, labels = get_model()

def infer_text(text):
    """Infer ESG category probabilities for a single text string."""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
    pred = {labels[i]: float(probs[0][i]) for i in range(len(labels))}
    return pred

# Example usage
if __name__ == "__main__":
    test_text = "Shell experienced an oil spill off the coast of Nigeria, resulting in a significant environmental fine."
    pred = infer_text(test_text)
    print("Input text:", test_text)
    print("Predicted ESG probabilities:", pred)