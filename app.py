import os

from flask import Flask, request, jsonify, render_template
from function4 import process_csv
from function import text_processing_model1, text_processing_model2, text_processing_model3, text_processing_model4, text_processing_model5, text_processing_model6

app = Flask(__name__)

# 存储上传文件的目录
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# 模拟文本处理函数
def process_file(selected_model, file_content):
    if selected_model == 'model1':
        processed_text = text_processing_model1(file_content)
    elif selected_model == 'model2':
        processed_text = text_processing_model2(file_content)
    elif selected_model == 'model3':
        processed_text = text_processing_model3(file_content)
    elif selected_model == 'model4':
        a = 1
    else:
        processed_text = "未选择有效的模型"

    return processed_text


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.get_json()
    selected_model = data.get('model', '')
    input_text = data.get('text', '')

    # 根据选择的模型和输入的文本进行处理
    processed_text = process_file(selected_model, input_text)

    return jsonify({'result': processed_text})


@app.route('/upload_file', methods=['POST'])
def upload_file():
    selected_model = request.form['selected_model']
    selected_function = request.form['extra_function']
    print(selected_model)
    file = request.files['file']
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        if selected_model == 'model4' and selected_function == 'function1':
            process_csv(file_path, 'after/out.txt')
            # 采用其他方式处理文件
            processed_text, plot_base64 = text_processing_model4(file_path,'after/out.txt')
            result_html = f"<div>{processed_text}</div><br><img src='data:image/png;base64,{plot_base64}'/>"
            return jsonify({'result': result_html})
        elif selected_model == 'model4' and selected_function == 'function2':
            process_csv(file_path, 'after/out.txt')
            # 采用其他方式处理文件
            processed_text, plot_base64 = text_processing_model5(file_path,'after/out.txt')
            result_html = f"<div>{processed_text}</div><br><img src='data:image/png;base64,{plot_base64}'/>"
            return jsonify({'result': result_html})
        elif selected_model == 'model4' and selected_function == 'function3':
            process_csv(file_path, 'after/out.txt')
            # 采用其他方式处理文件
            processed_text, plot_base64 = text_processing_model6(file_path,'after/out.txt')
            result_html = f"<div>{processed_text}</div><br><img src='data:image/png;base64,{plot_base64}'/>"
            return jsonify({'result': result_html})
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                processed_text = process_file(selected_model, file_content)
            return jsonify({'result': processed_text})
    else:
        return jsonify({'error': '未接收到文件'})


if __name__ == '__main__':
    app.run(debug=True)
