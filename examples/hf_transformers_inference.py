import requests
import json

from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import set_seed
# set_seed(42)

model_path_w16a16 = '/model/ModelScope/Qwen/Qwen3-0.6B'
model_path_w8a8_int8 = '/model/ModelScope/Qwen/Qwen3-0.6B'

common_eval_data = "datasets/common_eval_data.json"


def model_inference(model_path, eval_path = None, 
                    items = None, assistant_name = None) -> List[Dict]:

    llm = AutoModelForCausalLM.from_pretrained(model_path, 
                                                    torch_dtype=torch.float16,
                                                    device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    prompts = []
    if items == None:
        items = {}
        assert common_eval_data is not None
        list_data_dict = json.load(open(eval_path, "r"))
        for temp in list_data_dict:
            prompts.append(temp.get("user"))
    else: 
        for key, value in items.items():
            prompts.append(value.get("user"))

    print(prompts)
    input_ids = tokenizer(prompts, padding=True, return_tensors='pt').to("cuda")

    outputs = llm.generate(**input_ids,
                            max_length=256,
                            do_sample=True,
                            top_k=50,
                            temperature=0.7,
                            pad_token_id=tokenizer.eos_token_id)

    i = 0
    for prompt, output in zip(prompts, outputs):
        outputs_text = tokenizer.decode(output, skip_special_tokens=True)
        print(f"Prompt: {prompt}\n Assistant: {assistant_name}\n Generated text: {outputs_text}")
        key = "item_"+str(i)
        if items.get(key) == None:
            items[key] = {
                "user": prompt,
                assistant_name: outputs_text
            }
        else:
            temp = items.get(key)
            temp[assistant_name] = outputs_text
        i+=1

    return items

items = model_inference(model_path_w16a16, eval_path=common_eval_data, assistant_name="assistant_w16a16")
items = model_inference(model_path_w16a16, items=items, assistant_name="assistant_w8a8_int8")

results = []

for key, value in items.items():
    results.append(value)

result_path = "outputs/result.json"

with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

