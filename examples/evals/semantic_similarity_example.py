import textdistance
import json
from statistics import mean
from sentence_transformers import SentenceTransformer
import numpy as np

eval_path = "result.json"
list_data_dict = json.load(open(eval_path, "r"))

def levenshtein_assistant(list_data_dict, assistant_a, assistant_b, max_text_len=None):
    distances = []
    for temp in list_data_dict:
        if max_text_len is not None:
            distance = textdistance.levenshtein.distance(temp.get(assistant_a)[:max_text_len], temp.get(assistant_b)[:max_text_len])
        else:
            distance = textdistance.levenshtein.distance(temp.get(assistant_a)[:max_text_len], temp.get(assistant_b)[:max_text_len])
        distances.append(distance)
    print(f"编辑距离，最大文本长度限制为：{max_text_len}")
    print(distances)
    print(mean(distances))

levenshtein_assistant(list_data_dict, assistant_a="assistant_w16a16", assistant_b="assistant_w8a8_int8", max_text_len=10)
levenshtein_assistant(list_data_dict, assistant_a="assistant_w16a16", assistant_b="assistant_w8a8_int8", max_text_len=50)
levenshtein_assistant(list_data_dict, assistant_a="assistant_w16a16", assistant_b="assistant_w8a8_int8", max_text_len=100)
levenshtein_assistant(list_data_dict, assistant_a="assistant_w16a16", assistant_b="assistant_w8a8_int8", max_text_len=150)
levenshtein_assistant(list_data_dict, assistant_a="assistant_w16a16", assistant_b="assistant_w8a8_int8", max_text_len=200)
levenshtein_assistant(list_data_dict, assistant_a="assistant_w16a16", assistant_b="assistant_w8a8_int8")

def cosine_assistant(list_data_dict, assistant_a, assistant_b):
    cosines = []
    for temp in list_data_dict:
        similarity_cos = textdistance.cosine.similarity(temp.get(assistant_a), temp.get(assistant_b))
        cosines.append(similarity_cos)
    print("余弦相似度：")
    print(cosines)
    print(mean(cosines))

cosine_assistant(list_data_dict, assistant_a="assistant_w16a16", assistant_b="assistant_w8a8_int8")



def sentence_cosine(list_data_dict, assistant_a, assistant_b):
    embedding_path = "/root/models/Qwen3-Embedding-0.6B"
    # 如果启用 flash_attention_2 以更好的加速和节约内存
    model = SentenceTransformer(embedding_path, 
                                model_kwargs={"device_map": "auto"},
                                tokenizer_kwargs={"padding_side": "left"})

    assistant_a_list = []
    assistant_b_list = []
    for temp in list_data_dict:
        assistant_a_list.append(temp.get(assistant_a))
        assistant_b_list.append(temp.get(assistant_b))

    query_embeddings = model.encode(assistant_a_list)
    document_embeddings = model.encode(assistant_b_list)

    # Compute the (cosine) similarity between the query and document embeddings
    similarity = model.similarity(query_embeddings, document_embeddings)
    print("句向量+余弦相似度：")
    print(similarity)
    results = np.diag(similarity.numpy())
    print(results)
    result = np.mean(results)
    print(result)



sentence_cosine(list_data_dict, assistant_a="assistant_w16a16", assistant_b="assistant_w8a8_int8")
# sentence_cosine(list_data_dict, assistant_a="assistant_w16a16", assistant_b="assistant_w16a16")


