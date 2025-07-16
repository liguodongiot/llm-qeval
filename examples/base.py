from llmqeval.quantization.kernel import zeropoint_int8, absmax_int8
from llmqeval.evaluation.perplexity import calculate_perplexity
from llmqeval.visualization.layer import plot_weight
from llmqeval.utils.model import generate_text

from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy
import time

model_id = '/model/ModelScope/Qwen/Qwen3-0.6B'
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

view_linear = ["self_attn.q_proj", "self_attn.k_proj",
               "self_attn.v_proj", "self_attn.o_proj",
               "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

start = time.perf_counter()
for name, param in model.named_parameters():
    for linear in view_linear:
        if linear in name :
            weights_origin = param.cpu().detach().float().numpy()
            rows_origin, cols_origin = weights_origin.shape
            rows_origin, cols_origin
            sub_mat_size = 8
            weights = weights_origin.reshape(rows_origin//sub_mat_size, sub_mat_size, cols_origin//sub_mat_size, sub_mat_size).swapaxes(1, 2).reshape(-1, sub_mat_size, sub_mat_size).max(axis=(1, 2)).reshape(rows_origin//sub_mat_size,cols_origin//sub_mat_size)
            file_name = name.replace('.', '-')
            plot_weight(weights, f"{file_name}.pdf")

end = time.perf_counter()
exe_time = end - start
print(f"总执行时间：{exe_time:.6f} 秒")

# Generate text with original and quantized models
original_text = generate_text(model, "简单介绍下千问大模型", max_length=256)

print("-" * 50)
print(f"原始模型:\n{original_text}")
print("-" * 50)

long_text = "人工智能是一种变革性技术，正在重塑各行各业。"
ppl_original = calculate_perplexity(model, tokenizer, long_text)
print(f"\nPerplexity (原始模型): {ppl_original.item():.2f}")








