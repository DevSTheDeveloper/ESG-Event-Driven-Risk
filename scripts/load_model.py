# load_model.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Device setup
device = torch.device("mps") if torch.backends.mps.is_built() else torch.device("cpu")
print(f"Using device: {device}")

# Load model
MODEL_NAME = "yiyanghkust/finbert-esg-9-categories"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()
print("FinBERT 9-category ESG model loaded successfully")

# Function to return model objects
def get_model():
    return tokenizer, model, device

# Optional test
if __name__ == "__main__":
    sample_text = "Shell experienced an oil spill off the coast of Nigeria, resulting in a significant environmental fine."
    tokenizer, model, device = get_model()
    inputs = tokenizer(sample_text, return_tensors="pt").to(device)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    labels = model.config.id2label
    pred = {labels[i]: float(probs[0][i]) for i in range(len(labels))}
    print(f"Input text: {sample_text}")
    print(f"Predicted ESG probabilities: {pred}")