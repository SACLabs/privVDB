import os
from dotenv import load_dotenv
import numpy as np

import requests
requests.packages.urllib3.disable_warnings()
requests.packages.urllib3.util.ssl_.DEFAULT_CIPHERS += ':HIGH:!DH:!aNULL'
try:
    requests.packages.urllib3.contrib.pyopenssl.util.ssl_.DEFAULT_CIPHERS += ':HIGH:!DH:!aNULL'
except AttributeError:
    # no pyopenssl support used / needed / available
    pass
requests.packages.urllib3.util.ssl_.DEFAULT_CIPHERS += ':HIGH:!DH:!aNULL'
requests.packages.urllib3.disable_warnings()

if True:
    import openai


def openai_login(azure=False):
    load_dotenv()
    if azure is True:
        openai.api_key = os.getenv("AZURE_API_KEY")
        openai.api_base = os.getenv("AZURE_ENDPOINT")
        openai.api_type = "azure"
        openai.api_version = "2023-05-15"  # 请替换为您的 Azure OpenAI 服务的版本
    else:
        openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.verify_ssl_certs = False


openai_login(False)


def get_embeddings(text, embedding_type="openai"):
    if embedding_type == "fake":
        embed = list(np.random.normal(0, 0.02, size=(1536)))
        return {"embed": embed, "usage": 0}
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002",
        # deployment_id="textembedding"

    )
    # 打印生成的文本
    # print(response)
    embed = response["data"][0]["embedding"]
    # print(response)
    # print(embed)
    # print(len(embed))
    return {"embeddings": embed, "usage": response["usage"]["total_tokens"]}


if __name__ == "__main__":
    get_embeddings("QAQ!!")
    # response = openai.Embedding.create(
    #     model="text-embedding-ada-002", input=text)
    # # 从响应中提取textembedding，它是一个浮点数列表
    # embedding = response["embedding"]
    # # 打印textembedding
    # print(embedding)
