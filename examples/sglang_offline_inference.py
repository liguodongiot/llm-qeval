import requests
import json

import sglang as sgl

from sglang.srt.conversation import chat_templates
from sglang.test.test_utils import is_in_ci
from sglang.utils import async_stream_and_merge, stream_and_merge

llm = sgl.Engine(model_path="qwen/qwen2.5-0.5b-instruct")

common_eval_data = "datasets/common_eval_data.json"

list_data_dict = json.load(open(common_eval_data, "r"))

prompts = []
for temp in list_data_dict:
    prompts.append(temp)

sampling_params = {"temperature": 0.8, "top_p": 0.95}

outputs = llm.generate(prompts, sampling_params)

items = []
for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}\nGenerated text: {output['text']}")
    item = {
        "user": prompt, 
        "assistant_w16a16": output
    }
    items.append(item)

with open('output.json', 'w', encoding='utf-8') as f:
    json.dump(items, f)

