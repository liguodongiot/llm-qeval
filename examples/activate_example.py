from llmqeval.visualization.activate import plot_activate

from llmqeval.utils.model import generate_text, add_transformer_layer_hook, \
    add_antention_layer_hook, \
    add_mlp_layer_hook, \
    add_rmsnorm_layer_hook


from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy
import time

model_id = '/model/ModelScope/Qwen/Qwen3-0.6B'
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

# 用于存储每一层的输入和输出
activations = {}

# 定义钩子函数
def get_activation(name):
    def hook(module, input, output):
        # module：Qwen3DecoderLayer
        # input：<class 'tuple'> 1
        # output：<class 'tuple'> 1
        # print(module)
        # print(f"{type(module).__name__}")
        # print(type(input), len(input))
        # print(type(output), len(output))
        activations[name] = {
            'input': input[0].detach().cpu().numpy(),
            'output': output[0].detach().cpu().numpy()
        }
    return hook

add_transformer_layer_hook(model, get_activation)



# def get_attention_activation(name):
#     def hook(module, input, output):
#         # Qwen3Attention
#         # <class 'tuple'> 0
#         # <class 'tuple'> 2
#         print(module)
#         print(f"{type(module).__name__}")
#         print(type(input), len(input))
#         print(type(output), len(output))
#         print(output)
#         activations[name] = {
#             'output': output[0].detach().cpu().numpy()
#         }
#     return hook

# add_antention_layer_hook(model, get_attention_activation)

# def get_activation(name):
#     def hook(module, input, output):
#         # module：Qwen3MLP
#         # input：<class 'tuple'> 1
#         # output：<class 'torch.Tensor'> 1
#         # print(module)
#         # print(f"{type(module).__name__}")
#         # print(type(input), len(input))
#         # print(type(output), len(output))
#         activations[name] = {
#             'input': input[0].detach().cpu().numpy(),
#             'output': output.detach().cpu().numpy()
#         }
#     return hook

# add_mlp_layer_hook(model, get_activation)


# def get_activation(name):
#     def hook(module, input, output):
#         # module：Qwen3RMSNorm
#         # input：<class 'tuple'> 1
#         # output：<class 'torch.Tensor'> 1
#         print(module)
#         print(f"{type(module).__name__}")
#         print(type(input), len(input))
#         print(type(output), len(output))
#         activations[name] = {
#             'input': input[0].detach().cpu().numpy(),
#             'output': output.detach().cpu().numpy()
#         }
#     return hook

# add_rmsnorm_layer_hook(model, get_activation)


# 输入文本
input_text = """
Large language models (LLMs) show excellent performance but are compute- and memoryintensive. Quantization can reduce memory and
accelerate inference. However, existing methods cannot maintain accuracy and hardware efficiency at the same time. 
We propose SmoothQuant, a training-free, accuracy-preserving, and generalpurpose post-training quantization (PTQ) solution
to enable 8-bit weight, 8-bit activation (W8A8) quantization for LLMs. Based on the fact that
weights are easy to quantize while activations are not, SmoothQuant smooths the activation outliers
by offline migrating the quantization difficulty from activations to weights with a mathematically equivalent transformation. 
SmoothQuant enables an INT8 quantization of both weights and activations for all the matrix multiplications in
LLMs, including OPT, BLOOM, GLM, MT-NLG, Llama-1/2, Falcon, Mistral, and Mixtral models.
We demonstrate up to 1.56× speedup and 2× memory reduction for LLMs with negligible loss in accuracy. 
SmoothQuant enables serving 530B LLM within a single node. Our work offers a turn-key solution that reduces hardware costs and democratizes LLMs.
"""

# input_text = """
# Hello, how are you? 
# """
    
inputs = tokenizer(input_text, return_tensors='pt')

# 前向传播
model(inputs['input_ids'].cuda())

# print(activations)

# print(activations['layer_1']['output'])
# print(activations['layer_2']['input'])

start = time.perf_counter()
# 查看激活值
for name, activate_origin in activations.items():
    print(f"{name}:")
    print(f"Input shape: {activate_origin['input'].shape}")
    print(f"Output shape: {activate_origin['output'].shape}")

    # 三维变二维
    activate_input = activate_origin['input'].reshape(-1, len(activate_origin['input'][-1]))
    activate_strip = 2
    token_strip = 4
    activate = activate_input[::token_strip, ::activate_strip]
    plot_activate(activate, f"{name}_input_activate.pdf")

end = time.perf_counter()
exe_time = end - start
print(f"总执行时间：{exe_time:.6f} 秒")


