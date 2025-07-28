
from llmqeval.evaluation.perplexity import calculate_perplexity
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F


eval_path = "result.json"
list_data_dict = json.load(open(eval_path, "r"))

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
            
        # 计算每个位置的条件概率
        probs = F.softmax(logits, dim=-1)
        shifted_probs = probs[:, :-1, :]  # 移除最后一个预测
        shifted_tokens = inputs.input_ids[:, 1:]  # 对齐实际token
        
        # 聚合整个文本的分布
        vocab_probs = torch.zeros(tokenizer.vocab_size)
        for pos in range(shifted_tokens.shape[1]):
            token_id = shifted_tokens[0, pos].item()
            vocab_probs[token_id] += shifted_probs[0, pos, token_id].item()
        
        # 归一化得到全局分布
        return vocab_probs / vocab_probs.sum()

    # 获取两个文本的分布
    P = get_token_distribution(text1)
    Q = get_token_distribution(text2)
    
    # 计算KL散度 (P || Q)
    kl_div = F.kl_div(
        Q.log().unsqueeze(0), 
        P.unsqueeze(0), 
        # reduction='batch_mean',
        log_target=False
    )
    
    return kl_div.item()


text1 = "The quick brown fox"
text2 = "Jumps over the lazy dog"
kl_score = compute_kl_divergence(text1, text2, model_path = model_id)
print(f"KL(P || Q) = {kl_score:.4f}")

