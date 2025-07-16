
from llmqeval.quantization.kernel import zeropoint_int8, absmax_int8
from llmqeval.evaluation.perplexity import calculate_perplexity
from llmqeval.visualization.layer import plot_weight

from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy

model_id = '/model/ModelScope/Qwen/Qwen3-0.6B'
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

model_absmax = deepcopy(model)

for name, param in model_absmax.named_parameters():
    if "self_attn.q_proj" in name:
        _, dequantized = absmax_int8(param.data)
        param.data = dequantized

def generate_text(model, input_text, max_length=50):
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to("cuda")
    output = model.generate(inputs=input_ids,
                            max_length=max_length,
                            do_sample=True,
                            top_p=0.8,
                            pad_token_id=tokenizer.eos_token_id,
                            attention_mask=input_ids.new_ones(input_ids.shape))
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Generate text with original and quantized models
original_text = generate_text(model, "简单介绍下千问大模型", max_length=256)
absmax_text   = generate_text(model_absmax, "简单介绍下千问大模型", max_length=256)

print("-" * 50)
print(f"原始模型:\n{original_text}")
print("-" * 50)
print(f"Absmax量化模型:\n{absmax_text}")
print("-" * 50)


long_text = "人工智能是一种变革性技术，正在重塑各行各业。"

ppl_original = calculate_perplexity(model, tokenizer, long_text)
ppl_absmax = calculate_perplexity(model_absmax, tokenizer, long_text)

print(f"\nPerplexity (原始模型): {ppl_original.item():.2f}")
print(f"Perplexity (Absmax量化模型): {ppl_absmax.item():.2f}")




