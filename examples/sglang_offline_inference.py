import requests
import json

import sglang as sgl

from sglang.srt.conversation import chat_templates
from sglang.test.test_utils import is_in_ci
from sglang.utils import async_stream_and_merge, stream_and_merge
from typing import Dict, List


model_path_w16a16 = '/model/ModelScope/Qwen/Qwen3-0.6B'
model_path_w8a8_int8 = '/model/ModelScope/Qwen/Qwen3-0.6B'

common_eval_data = "datasets/common_eval_data.json"


def model_inference(model_path, eval_path = None, 
                    items = None, assistant_name = None,
                    result_path = "result.json") -> List[Dict]:
    llm = sgl.Engine(model_path=model_path)

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

    sampling_params = {"temperature": 0, "top_k": 1, "seed": 66}
    outputs = llm.generate(prompts, sampling_params)

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

