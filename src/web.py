# 导入Flask模块
from flask import Flask, render_template, request
import milvus
import numpy as np
# 创建一个Flask应用
app = Flask(__name__)

# 定义一个转换函数，根据你的需求实现转换逻辑


def insert_dp_text_and_embeddings(text):
    try:
        dp_text = privDB.get_dp_text("admin", text, 3)
        embeds = privDB.insert_text(dp_text)
        embeds = str(np.array(embeds))
    except Exception as e:
        dp_text = "ERROR"
        embeds = str(e)

    return dp_text, embeds

# 定义一个路由，用来处理网页的请求

# 定义一个生成列表的函数，根据你的需求实现生成逻辑


def get_search_result(text):
    try:
        lis = privDB.search_text(text)
        lis = [(i, x[0], x[1]) for i, x in enumerate(zip(lis[0], lis[1]))]
    except Exception as e:
        lis = [(str(e), 0)]
    return lis

# 定义一个新的路由，用来处理生成列表的请求


@app.route('/', methods=['GET', 'POST'])
def index():
    # 如果是GET请求，就渲染一个表单页面，让用户输入文本
    if request.method == 'GET':
        return render_template('form.html')
    # 如果是POST请求，就获取用户输入的文本，调用转换函数，渲染一个结果页面，显示转换后的文本和数组
    elif request.method == 'POST':
        import logging
        logging.warning(request.form.to_dict())
        button = request.form.get('btn')
        logging.warning(button)
        if button == "1":
            text = request.form.get('text')
            converted_text, converted_array = insert_dp_text_and_embeddings(
                text)
            logging.info(converted_text)
            return render_template('result.html', text=text, converted_text=converted_text, converted_array=converted_array)
        elif button == "2":
            text = request.form.get('text')
            list_ = get_search_result(text)
            return render_template('list.html', text=text, list_=list_)
        elif button == "3":
            privDB.init_database()
            return render_template('form.html')
        else:
            logging.info(button)


# 运行Flask应用
if __name__ == '__main__':
    config = {}
    dp_config = {
        "base_dataset_path": """D:\Codes\pjlab\\vdb\privVDB\\v0.1\privDB\data\CBTest\data\cbt_train.txt""",
        "embedding_type": "glove",
        "non_sensitive_p": 0.3,
        "sensitive_word_percentage": 0.9,
        "epsilons": [1, 3],
        "DP_mech": "base"
    }
    config["text_dp_type"] = "santext"
    config["text_dp_config"] = dp_config
    privDB = milvus.privVDB(config)
    app.run(debug=True)
