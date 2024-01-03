# 导入Flask模块
from flask import Flask, render_template, request
import milvus
import numpy as np
import argparse

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Description of your program.")

    parser.add_argument("--base_dataset_path", type=str,
                        help="Path to cbt_train.txt")

    parser.add_argument("--embedding_type", type=str, default="glove",
                        help="Type of embedding.")

    parser.add_argument("--non_sensitive_p", type=float, default=0.3,
                        help="Non-sensitive parameter.")

    parser.add_argument("--sensitive_word_percentage", type=float, default=0.9,
                        help="Percentage of sensitive words.")

    parser.add_argument("--epsilons", type=int, nargs="+", default=[1, 3],
                        help="List of epsilon values.")

    parser.add_argument("--dp_mech", type=str, default="base",
                        help="Differential Privacy mechanism.")

    parser.add_argument("--embedding_path", type=str,
                        help="Path to glove.840B.300d.txt.")

    parser.add_argument("--text_dp_type", type=str, default="santext",
                        help="Type of text differential privacy.")

    return parser.parse_args()


# 运行Flask应用
if __name__ == '__main__':
    args = parse_args()
    config = {}
    dp_config = {
        "base_dataset_path": args.base_dataset_path,
        "embedding_type": args.embedding_type,
        "embedding_path": args.embedding_path,
        "non_sensitive_p": args.non_sensitive_p,
        "sensitive_word_percentage": args.sensitive_word_percentage,
        "epsilons": args.epsilons,
        "DP_mech": args.dp_mech,
        "args": args
    }
    config["text_dp_type"] = args.text_dp_type
    config["text_dp_config"] = dp_config
    privDB = milvus.PrivVDB(config)
    # privDB = "1"
    print("start")
    app.run(debug=False, port=6001)
