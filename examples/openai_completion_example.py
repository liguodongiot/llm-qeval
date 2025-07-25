
from openai import OpenAI

def text_generate(base_url="http://127.0.0.1:8000/v1/", model_name="qwen2.5-32b", query=""):

    client_qwen = OpenAI(
        api_key="",
        base_url=base_url  # 模型地址
    )

    try:
        chat_completion = client_qwen.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": query,
                }
            ],
            model=model_name,  # 此处可更换其它模型
            stream=False,
            temperature=0,
            max_tokens=1024
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(e)
        return "request_limit_reached"


base_url = "http://10.1.x.1:8225/v1/"
model_name = "qwen2.5-32b"
query = """字节跳动的企业文化是什么？"""

print(text_generate(query, model_name, query))


