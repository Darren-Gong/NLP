import paddle

from function1 import load_data, build_dict, text_to_input, MyLSTM


def predict_label(test_text):
    # 加载训练数据和构建字典
    train_data = load_data('../model/Text Classification/data/new.txt', False)
    word_idx, label_idx = build_dict(train_data)

    # 创建模型实例并加载预训练权重
    model = MyLSTM()
    model_state_dict = paddle.load('../model/Text Classification/result/model_final1.pdparams')
    model.set_state_dict(model_state_dict)

    # 将文本转换为模型输入格式
    input_seq = text_to_input(test_text, word_idx)
    input_tensor = paddle.to_tensor(input_seq, dtype='int32').unsqueeze(0)

    # 使用模型进行预测
    model.eval()
    with paddle.no_grad():
        output = model(input_tensor)
        predicted_label_idx = paddle.argmax(output, axis=1).item()

    inverted_dict = {v: k for k, v in label_idx.items()}
    predicted_label = inverted_dict[predicted_label_idx]
    return predicted_label


if __name__ == "__main__":
    # 测试函数
    test_text = "96年最强点今成大隐患 姚明生不逢时无缘最高待遇"
    predicted_label = predict_label(test_text)
    # print("Predicted Label:", predicted_label)
