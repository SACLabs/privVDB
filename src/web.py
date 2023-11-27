# 导入Flask模块
from flask import Flask, render_template, request

# 创建一个Flask应用
app = Flask(__name__)

# 定义一个转换函数，根据你的需求实现转换逻辑


def convert(text):
    # 这里是示例代码，你可以根据你的需求修改
    # 转换后的文本是原文本的倒序
    converted_text = text[::-1]
    # 转换后的数组是原文本中每个字符的ASCII码
    converted_array = [ord(c) for c in text]
    # 返回转换后的文本和数组
    return converted_text, converted_array

# 定义一个路由，用来处理网页的请求

# 定义一个生成列表的函数，根据你的需求实现生成逻辑


def generate_list(text):
    # 这里是示例代码，你可以根据你的需求修改
    # 生成的列表是原文本按空格分割后的子串，每个子串的得分是它的长度
    lis = []
    for word in text.split():
        lis.append((word, len(word)))
    import time
    time.sleep(3)
    # 返回生成的列表
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
        logging.warning(request.form.get('insert'))
        button = request.form.get('insert') or request.form.get('search')
        logging.warning(button)
        if button == "1":
            text = request.form.get('text')
            converted_text, converted_array = convert(text)
            return render_template('result.html', text=text, converted_text=converted_text, converted_array=converted_array)
        else:
            text = request.form.get('text')
            list_ = generate_list(text)
            return render_template('list.html', text=text, list_=list_)


# 运行Flask应用
if __name__ == '__main__':
    app.run(debug=True)
