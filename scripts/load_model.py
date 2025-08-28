# scripts/load_model.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Use MPS if available, otherwise CPU
device = torch.device("mps") if torch.backends.mps.is_built() else torch.device("cpu")
print(f"Using device: {device}")

MODEL_NAME = "yiyanghkust/finbert-esg-9-categories"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()
print("FinBERT 9-category ESG model loaded successfully")

# Category labels (should match model's output)
labels = [
    "Climate Change", "Natural Capital", "Pollution & Waste",
    "Human Capital", "Product Liability", "Community Relations",
    "Corporate Governance", "Business Ethics & Values", "Non-ESG"
]

def get_model():
    return tokenizer, model, device, labels

# Test function
if __name__ == "__main__":
    test_text = "Shell experienced an oil spill off the coast of Nigeria, resulting in a significant environmental fine."
    tokenizer, model, device, labels = get_model()
    inputs = tokenizer(test_text, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
    pred = {labels[i]: float(probs[0][i]) for i in range(len(labels))}
    print("Input text:", test_text)
    print("Predicted ESG probabilities:", pred)