
from llmqeval.evaluation.perplexity import calculate_perplexity
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import numpy as np


model_id = '/model/ModelScope/Qwen/Qwen3-0.6B'

def model_ppl(model_id):
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    long_text = "当客户在金融机构申请贷款时，应注意以下几个方面以确保自己的权益和财务安全：\n\n1. **了解贷款产品**：在申请贷款前，应详细了解贷款产品的相关信息，包括贷款额度、利率、还款方式、期限等。确保自己对贷款产品有充分的了解。\n\n2. **比较不同产品**：不要急于决定，可以比较不同金融机构提供的贷款产品，选择最适合自己的贷款方案。\n\n3. **注意隐藏费用**：除了利率外，还应注意是否有其他费用，如手续费、提前还款费等。这些费用可能会增加贷款成本。\n\n4. **评估还款能力**：在申请贷款前，应评估自己的还款能力，确保自己能够按时还款，避免因无法按时还款而产生额外费用或影响个人信用记录。\n\n5. **保护个人信息**：在申请贷款过程中，应注意保护个人信息安全，避免个人信息泄露。只向正规金融机构提供个人信息。\n\n6. **阅读合同条款**：在签署贷款合同前，务必仔细阅读合同条款，确保自己理解所有条款内容。如有疑问，应及时向金融机构咨询或寻求法律建议。\n\n7. **了解提前还款政策**：如果有可能提前还款，应了解提前还款的政策和可能产生的费用，以便做出最有利的选择。\n\n通过以上几点，可以帮助客户在申请金融机构的贷款时做出更加明智的决定，保护自己的权益。"
    ppl_original = calculate_perplexity(model, tokenizer, long_text)

    print(f"\nPerplexity (原始模型): {ppl_original.item():.2f}")
    del model
    del tokenizer

# model_ppl()


def compute_kl_divergence(text1, text2, model_path="gpt2"):
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    
    
    # 获取文本的token概率分布
    def get_token_distribution(text):
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            logits = outputs.logits
        
        epsilon=1e-10
        # 计算每个位置的条件概率
        probs = F.softmax(logits, dim=-1)

        llm_probs = probs[:, :, :]  # 移除最后一个预测
        llm_tokens = inputs.input_ids[:, :]  # 对齐实际token
        
        # 初始化词汇表分布（使用对数空间）
        vocab_log_probs = torch.full((tokenizer.vocab_size,), -1e10)  # 极小值代替负无穷
        
        # 聚合整个文本的log概率
        for pos in range(llm_tokens.shape[1]):
            token_id = llm_tokens[0, pos].item()
            token_log_prob = llm_probs[0, pos, token_id].item()
            
            # 使用logsumexp进行数值稳定的聚合
            if vocab_log_probs[token_id] < -1e9:  # 第一次出现
                vocab_log_probs[token_id] = token_log_prob
            else:
                # 对数空间的概率聚合
                vocab_log_probs[token_id] = np.logaddexp(
                    vocab_log_probs[token_id], token_log_prob
                )
        
        # 转换为概率分布并添加平滑
        vocab_probs = F.softmax(vocab_log_probs, dim=0).numpy()
        vocab_probs = (1 - epsilon) * vocab_probs + epsilon / tokenizer.vocab_size
        
        # 验证归一化
        assert abs(vocab_probs.sum() - 1.0) < 1e-5, "概率分布未正确归一化"
        return vocab_probs
    
    try:
        # 获取两个文本的分布
        P = get_token_distribution(text1)
        Q = get_token_distribution(text2)
        
        # 计算KL散度 (P || Q)，避免除零错误
        kl_div = 0.0
        for i in range(len(P)):
            if P[i] > 0 and Q[i] > 0:
                kl_div += P[i] * (np.log(P[i]) - np.log(Q[i]))
        
        return kl_div
    
    except Exception as e:
        print(f"计算错误: {e}")
        return float('nan')

text1 = "The quick brown fox jumps over the lazy dog"
text2 = "A fast auburn fox leaps above a sleepy hound"
kl_score = compute_kl_divergence(text1, text2, model_id)

if not np.isnan(kl_score):
    print(f"KL(P || Q) = {kl_score:.4f}")
else:
    print("KL散度计算失败，请检查输入文本和模型")