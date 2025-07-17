
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from llmqeval.evaluation.accuracy import Evaluator
from llmqeval.quantization.model.qwen import quantize_qwen3

model_id = '/model/ModelScope/Qwen/Qwen3-0.6B'
model_fp16 = AutoModelForCausalLM.from_pretrained(model_id, 
                                                  torch_dtype=torch.float16,
                                                  device_map="auto")

tokenizer = AutoModelForCausalLM.from_pretrained(model_id)
# dataset = load_dataset("lambada", split="validation[:1000]")
dataset = load_dataset("json", data_files="datasets/lambada_validation_data_1k.json", split="train")
evaluator = Evaluator(dataset, tokenizer, "cuda")

acc_fp16 = evaluator.evaluate(model_fp16)
print(f"Original model (fp16) accuracy: {acc_fp16}")

model_w8a8 = quantize_qwen3(model_fp16)
print(model_w8a8)

acc_w8a8 = evaluator.evaluate(model_w8a8)
print(f"Naive W8A8 quantized model accuracy: {acc_w8a8}")

