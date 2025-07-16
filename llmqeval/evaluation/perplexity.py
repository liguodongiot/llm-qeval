
import torch

# Perplexity evaluation
def calculate_perplexity(model, tokenizer, text):
    encodings = tokenizer(text, return_tensors='pt').to("cuda")
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    return torch.exp(outputs.loss)


