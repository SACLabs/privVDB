# Private Vector Database

### 安装

1. 克隆仓库：`https://github.com/tc2000731/privVDB.git`
2. 安装依赖：详见`requirements.txt`
3. 安装milvus, 建议采用[docker ](https://milvus.io/docs/install_standalone-docker.md)

4.  启动milvus, 默认会监听19530端口

5. 下载[Glove6B300d](https://nlp.stanford.edu/data/glove.6B.zip), 把`src/desensitization.py` 中的word_embedding_path替换为glove.840B.300d.txt的位置

6. 下载[CBTest](https://drive.google.com/drive/folders/1K09Cg6IrgGfKgrAQodIJd7iE-oyMpk-a?usp=sharing), 把base_dataset_path替换为cbt_train.txt

7. 在根目录下添加个`.env`文件:
   ```
   OPENAI_API_KEY = sk-***
   # optional
   AZURE_API_KEY = ea***  
   AZURE_ENDPOINT = "https://test1115.openai.azure.com/" 
   ```

   



### 使用

##### 单独使用文本脱敏

```bash
python src/desensitization.py --base_dataset_path <path_to_cbt_train.txt> --embedding_path <path_to_glove.840B.300d.txt>
```

##### 使用web

```bash
python src/web.py --base_dataset_path <path_to_cbt_train.txt> --embedding_path <path_to_glove.840B.300d.txt>
```

##### 仅使用VDB

关于如何使用为文本脱敏VDB写死的class的一些示例:

```bash
python src/test_privVDB.py --base_dataset_path <path_to_cbt_train.txt> --embedding_path <path_to_glove.840B.300d.txt>
```

关于如何使用原始VDBhandler的一些示例

```bash
python src/test_privVDB.py
```

