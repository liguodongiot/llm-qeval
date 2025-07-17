

accuraty_example.py

```
load model.
load dataset.
Map: 100%|███████████████████████████████████████████████| 1000/1000 [00:00<00:00, 6911.44 examples/s]
model evaluate.
Original model (fp16) accuracy: 0.629
Qwen3ForCausalLM(
  (model): Qwen3Model(
    (embed_tokens): Embedding(151936, 1024)
    (layers): ModuleList(
      (0-27): 28 x Qwen3DecoderLayer(
        (self_attn): Qwen3Attention(
          (q_proj): W8A8Linear(1024, 2048, bias=False, weight_quant=per_channel, act_quant=per_token, output_quant=None)
          (k_proj): W8A8Linear(1024, 1024, bias=False, weight_quant=per_channel, act_quant=per_token, output_quant=None)
          (v_proj): W8A8Linear(1024, 1024, bias=False, weight_quant=per_channel, act_quant=per_token, output_quant=None)
          (o_proj): W8A8Linear(2048, 1024, bias=False, weight_quant=per_channel, act_quant=per_token, output_quant=None)
          (q_norm): Qwen3RMSNorm((128,), eps=1e-06)
          (k_norm): Qwen3RMSNorm((128,), eps=1e-06)
        )
        (mlp): Qwen3MLP(
          (gate_proj): W8A8Linear(1024, 3072, bias=False, weight_quant=per_channel, act_quant=per_token, output_quant=None)
          (up_proj): W8A8Linear(1024, 3072, bias=False, weight_quant=per_channel, act_quant=per_token, output_quant=None)
          (down_proj): W8A8Linear(3072, 1024, bias=False, weight_quant=per_channel, act_quant=per_token, output_quant=None)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
        (post_attention_layernorm): Qwen3RMSNorm((1024,), eps=1e-06)
      )
    )
    (norm): Qwen3RMSNorm((1024,), eps=1e-06)
    (rotary_emb): Qwen3RotaryEmbedding()
  )
  (lm_head): Linear(in_features=1024, out_features=151936, bias=False)
)
Naive W8A8 quantized model accuracy: 0.606
```