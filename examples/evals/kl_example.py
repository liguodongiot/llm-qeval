import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
from scipy.stats import entropy

# 配置全局字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['font.size'] = 12


# 设置美观的绘图样式
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)

def get_token_distribution(text, model_name="gpt2", epsilon=1e-10):
    """获取文本的token概率分布"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        logits = outputs.logits
    
    # 计算每个位置的条件概率
    log_probs = F.log_softmax(logits, dim=-1)

    shifted_log_probs = log_probs[:, :-1, :]
    shifted_tokens = inputs.input_ids[:, 1:]
    
    # 初始化词汇表分布
    vocab_log_probs = torch.full((tokenizer.vocab_size,), -1e10)
    
    # 聚合整个文本的log概率
    for pos in range(shifted_tokens.shape[1]):
        token_id = shifted_tokens[0, pos].item()
        token_log_prob = shifted_log_probs[0, pos, token_id].item()
        
        if vocab_log_probs[token_id] < -1e9:
            vocab_log_probs[token_id] = token_log_prob
        else:
            vocab_log_probs[token_id] = np.logaddexp(
                vocab_log_probs[token_id], token_log_prob
            )
    
    # 转换为概率分布并添加平滑
    vocab_probs = F.softmax(vocab_log_probs, dim=0).numpy()
    vocab_probs = (1 - epsilon) * vocab_probs + epsilon / tokenizer.vocab_size
    vocab_probs /= vocab_probs.sum()  # 确保归一化
    
    return vocab_probs, tokenizer

def compute_kl_divergence(P, Q):
    """计算KL散度"""
    kl_div = 0.0
    for i in range(len(P)):
        if P[i] > 0 and Q[i] > 0:
            kl_div += P[i] * (np.log(P[i]) - np.log(Q[i]))
    return kl_div

def visualize_distributions(P, Q, tokenizer, title="概率分布对比"):
    """可视化两个概率分布及其差异"""
    # 获取前20个最高概率的token
    top_n = 20
    top_indices = np.argsort(P)[-top_n:]
    
    # 准备数据
    tokens = [tokenizer.decode([idx]) for idx in top_indices]
    p_values = P[top_indices]
    q_values = Q[top_indices]
    contributions = [p * (np.log(p) - np.log(q)) if p > 0 and q > 0 else 0 
                    for p, q in zip(p_values, q_values)]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # 分布对比图
    x = np.arange(len(tokens))
    width = 0.35
    rects1 = ax1.bar(x - width/2, p_values, width, label='分布 P', color='skyblue')
    rects2 = ax1.bar(x + width/2, q_values, width, label='分布 Q', color='salmon')
    
    ax1.set_title(f'{title} (KL散度: {compute_kl_divergence(P, Q):.4f})')
    ax1.set_ylabel('概率')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tokens, rotation=45, ha='right')
    ax1.legend()
    
    # 添加数据标签
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax1.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 垂直偏移
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    # KL贡献图
    colors = ['red' if c > 0 else 'green' for c in contributions]
    ax2.bar(tokens, contributions, color=colors)
    ax2.set_title('KL散度贡献值')
    ax2.set_ylabel('贡献值')
    ax2.set_xticklabels(tokens, rotation=45, ha='right')
    ax2.axhline(0, color='black', linewidth=0.8)
    
    # 添加贡献值标签
    for i, v in enumerate(contributions):
        ax2.text(i, v + 0.001 if v >= 0 else v - 0.002, 
                f'{v:.4f}', 
                ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    # plt.show()
    plt.savefig("kl.pdf", dpi=100)


def visualize_kl_heatmap(texts, model_name="gpt2"):
    """创建文本对KL散度的热力图"""
    n = len(texts)
    kl_matrix = np.zeros((n, n))
    
    # 计算所有文本对的KL散度
    distributions = []
    tokenizers = []
    for text in texts:
        dist, tokenizer = get_token_distribution(text, model_name)
        distributions.append(dist)
        tokenizers.append(tokenizer)
    
    for i in range(n):
        for j in range(n):
            if i != j:
                kl_matrix[i][j] = compute_kl_divergence(distributions[i], distributions[j])
    
    # 创建热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(kl_matrix, annot=True, fmt=".4f", cmap="YlOrRd",
                xticklabels=[f"文本 {i+1}" for i in range(n)],
                yticklabels=[f"文本 {i+1}" for i in range(n)])
    
    plt.title("文本对KL散度热力图")
    plt.xlabel("分布 Q")
    plt.ylabel("分布 P")
    plt.show()
    
    return kl_matrix

def visualize_kl_over_time(text, model_name="gpt2", window_size=3):
    """可视化文本中KL散度随时间的变化"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    
    inputs = tokenizer(text, return_tensors="pt")
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    
    kl_values = []
    positions = []
    
    # 滑动窗口计算KL散度
    for i in range(1, len(tokens) - window_size):
        window_start = max(0, i - window_size)
        window_text = tokenizer.decode(inputs.input_ids[0][window_start:i+1])
        next_token = tokenizer.decode(inputs.input_ids[0][i+1])
        
        # 获取当前上下文分布
        current_dist, _ = get_token_distribution(window_text, model_name)
        
        # 获取下一个token的理想分布
        with torch.no_grad():
            outputs = model(**tokenizer(window_text, return_tensors="pt"))
            logits = outputs.logits[0, -1]
            next_token_probs = F.softmax(logits, dim=-1).numpy()
        
        # 计算KL散度
        kl_div = entropy(current_dist, next_token_probs) if entropy is not None else 0
        kl_values.append(kl_div)
        positions.append(i)
    
    # 创建图表
    plt.figure(figsize=(14, 6))
    plt.plot(positions, kl_values, marker='o', linestyle='-', color='purple')
    
    # 标记关键点
    max_kl_idx = np.argmax(kl_values)
    # plt.annotate(f"最大KL: {kl_values[max_kl_idx]:.4f}\nToken: {tokens[positions[max_kl_idx]+1}",
    plt.annotate(f"最大KL",
                xy=(positions[max_kl_idx], kl_values[max_kl_idx]),
                xytext=(positions[max_kl_idx]-2, kl_values[max_kl_idx]+0.05),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=10)
    
    plt.title("KL散度随时间变化 (滑动窗口分析)")
    plt.xlabel("Token位置")
    plt.ylabel("KL散度")
    plt.xticks(positions, tokens[1:-window_size], rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def visualize_kl_components(P, Q, tokenizer, top_n=15):
    """可视化KL散度的主要组成部分"""
    # 计算每个token的贡献
    contributions = []
    for i in range(len(P)):
        if P[i] > 0 and Q[i] > 0:
            contrib = P[i] * (np.log(P[i]) - np.log(Q[i]))
            contributions.append((i, contrib))
    
    # 按贡献值排序
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    top_contrib = contributions[:top_n]
    
    # 准备数据
    tokens = [tokenizer.decode([idx]) for idx, _ in top_contrib]
    contrib_values = [contrib for _, contrib in top_contrib]
    p_values = [P[idx] for idx, _ in top_contrib]
    q_values = [Q[idx] for idx, _ in top_contrib]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 8))
    index = np.arange(len(tokens))
    bar_width = 0.35
    
    # 绘制贡献值
    bars = ax.bar(index, contrib_values, bar_width, color='mediumpurple', label='KL贡献值')
    
    # 添加概率对比
    ax2 = ax.twinx()
    ax2.plot(index, p_values, 'o-', color='darkgreen', label='P概率')
    ax2.plot(index, q_values, 's--', color='darkred', label='Q概率')
    
    # 设置标签和标题
    ax.set_xlabel('Token')
    ax.set_ylabel('KL贡献值', color='mediumpurple')
    ax2.set_ylabel('概率值', color='black')
    ax.set_title('KL散度主要组成部分分析')
    ax.set_xticks(index)
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    
    # 添加数据标签
    for i, v in enumerate(contrib_values):
        ax.text(i, v + 0.001 if v >= 0 else v - 0.002, 
               f'{v:.4f}', 
               ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
    
    for i, (p, q) in enumerate(zip(p_values, q_values)):
        ax2.text(i, p + 0.01, f'{p:.3f}', ha='center', va='bottom', color='darkgreen', fontsize=9)
        ax2.text(i, q - 0.02, f'{q:.3f}', ha='center', va='top', color='darkred', fontsize=9)
    
    # 添加图例
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines, labels, loc='upper left')
    ax2.legend(lines2, labels2, loc='upper right')
    
    plt.tight_layout()
    plt.show()

# 示例使用
if __name__ == "__main__":

    # 示例文本
    text1 = "The quick brown fox jumps over the lazy dog"
    text2 = "A fast auburn fox leaps above a sleepy hound"

    text3 = "Artificial intelligence is transforming the world"
    text4 = "Machine learning algorithms are changing our lives"
    model_name = '/model/ModelScope/Qwen/Qwen3-0.6B'


    # 获取分布
    P, tokenizer = get_token_distribution(text1, model_name=model_name)

    Q, _ = get_token_distribution(text2, model_name=model_name)
    
    print("可视化1: 分布对比与KL贡献")

    visualize_distributions(P, Q, tokenizer, "狐狸与猎犬的文本分布")
    


    # print("\n可视化2: 文本对KL散度热力图")
    # texts = [text1, text2, text3, text4]
    # kl_matrix = visualize_kl_heatmap(texts)
    
    # print("\n可视化3: KL散度随时间变化")
    # long_text = "Natural language processing is a subfield of artificial intelligence that focuses on the interaction between computers and human language."
    # visualize_kl_over_time(long_text, window_size=5)
    
    # print("\n可视化4: KL散度组成分析")
    # visualize_kl_components(P, Q, tokenizer)