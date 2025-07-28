import requests
import json

from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch



model_path_w16a16 = '/model/ModelScope/Qwen/Qwen3-0.6B'
model_path_w8a8_int8 = '/model/ModelScope/Qwen/Qwen3-0.6B'

common_eval_data = "datasets/common_eval_data.json"


def model_inference(model_path, eval_path = None, 
                    items = None, assistant_name = None,
                    result_path = "result.json") -> List[Dict]:

    model_id = '/model/ModelScope/Qwen/Qwen3-0.6B'
    llm = AutoModelForCausalLM.from_pretrained(model_path, 
                                                    torch_dtype=torch.float16,
                                                    device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    prompts = []
    if items == None or len(items)==0:
        items = []
        assert common_eval_data is not None
        list_data_dict = json.load(open(eval_path, "r"))
        for temp in list_data_dict:
            prompts.append(temp)
    else:   
        for temp in items:
            prompts.append(temp)

    input_ids = tokenizer.encode(prompts, return_tensors='pt').to("cuda")
    outputs = llm.generate(inputs=input_ids,
                            max_length=256,
                            do_sample=True,
                            top_k=1,
                            pad_token_id=tokenizer.eos_token_id,
                            attention_mask=input_ids.new_ones(input_ids.shape))


    for prompt, output in zip(prompts, outputs):
        print(f"Prompt: {prompt}\n Assistant: {assistant_name}\n Generated text: {output['text']}")
        
        item = {
            "user": prompt, 
            assistant_name: output
        }
        
        items.append(item)

    
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(items, f)

    llm.shutdown()

    return items


items = model_inference(model_path_w16a16, eval_path=common_eval_data, assistant="assistant_w16a16")
items = model_inference(model_path_w8a8_int8, items=items, assistant="assistant_w8a8_int8")

