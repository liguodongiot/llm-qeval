
from typing import Callable, Optional, Union

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

def generate_text(model, tokenizer, input_text, max_length=50):
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to("cuda")
    output = model.generate(inputs=input_ids,
                            max_length=max_length,
                            do_sample=True,
                            top_p=0.8,
                            pad_token_id=tokenizer.eos_token_id,
                            attention_mask=input_ids.new_ones(input_ids.shape))
    return tokenizer.decode(output[0], skip_special_tokens=True)


def add_transformer_layer_hook(model, hook_function):
    # 为每一层注册钩子
    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(hook_function(f'layer_{i}'))
        break


def add_transformer_linear_layer_hook(model, hook_function):
    # 为每一层注册钩子
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.q_proj.register_forward_hook(hook_function(f'layer_{i}_self_attn_q_proj'))
        layer.self_attn.k_proj.register_forward_hook(hook_function(f'layer_{i}_self_attn_k_proj'))
        layer.self_attn.v_proj.register_forward_hook(hook_function(f'layer_{i}_self_attn_v_proj'))
        layer.self_attn.o_proj.register_forward_hook(hook_function(f'layer_{i}_self_attn_o_proj'))
        layer.mlp.gate_proj.register_forward_hook(hook_function(f'layer_{i}_mlp_gate_proj'))
        layer.mlp.up_proj.register_forward_hook(hook_function(f'layer_{i}_mlp_up_proj'))
        layer.mlp.down_proj.register_forward_hook(hook_function(f'layer_{i}_mlp_down_proj'))
        break

def add_rmsnorm_layer_hook(model, hook_function):
     for i, layer in enumerate(model.model.layers):
        layer.input_layernorm.register_forward_hook(hook_function(f'layer_{i}_input_layernorm'))
        layer.post_attention_layernorm.register_forward_hook(hook_function(f'layer_{i}_post_attention_layernorm'))
        break


def add_antention_layer_hook(model, hook_function):
     for i, layer in enumerate(model.model.layers):
        layer.self_attn.register_forward_hook(hook_function(f'layer_{i}_self_attn'))
        break


def add_mlp_layer_hook(model, hook_function):
     for i, layer in enumerate(model.model.layers):
        layer.mlp.register_forward_hook(hook_function(f'layer_{i}_mlp'))
        break





