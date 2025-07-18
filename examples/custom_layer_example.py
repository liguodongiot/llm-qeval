
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from torch import nn

from copy import deepcopy
import time

model_id = '/model/ModelScope/Qwen/Qwen3-0.6B'
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print(model)
print("---------------")

for name, module in model.named_modules():
    print(name, module)
print("---------------")


for name, param in model.named_parameters():
    print(name, param.size())

print("---------------")

activates = {}

class AttentionCatcher(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        # self.name = name

    def forward(self, **kwargs):
        print(kwargs)
        print("attention:")
        hidden_states = kwargs["hidden_states"]
        print(hidden_states)
        # activates[self.name] = hidden_states
        return self.module.forward(**kwargs)


class MlpCatcher(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        # self.name = name

    def forward(self, hidden_states):
        print("mlp:")
        print(hidden_states)
        return self.module.forward(hidden_states)

for i, layer in enumerate(model.model.layers):
    # layer.self_attn = Catcher(layer.self_attn, f"layer_{i}_self_attn")
    # layer.mlp = Catcher(layer.mlp, f"layer_{i}_mlp")
    # layer.self_attn = Catcher(layer.self_attn)
    layer.self_attn = AttentionCatcher(layer.self_attn)
    layer.mlp = MlpCatcher(layer.mlp)

print(model)

input_text = """
Hello, how are you? 
"""
    
inputs = tokenizer(input_text, return_tensors='pt')

# 前向传播
model(inputs['input_ids'].cuda())



# Generate text with original and quantized models
# original_text = generate_text(model, tokenizer,  "简单介绍下千问大模型", max_length=256)

# print("-" * 50)
# print(f"原始模型:\n{original_text}")

