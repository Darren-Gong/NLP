import collections
import torch
import paddle
import paddle.nn as nn
# 这个是文本分类的具体实现代码

def load_data(data_path, is_test=False):
    dataset = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not is_test:
                items = line.strip().split('\t')
                if len(items) != 3:
                    continue
                sent = items[2].strip()
                label = items[1].strip()
                dataset.append((sent, label))
            else:
                dataset.append(line.strip())
    return dataset


def build_dict(train_data):
    word_freq = collections.defaultdict(int)
    label_set = set()
    for seq, label in train_data:
        for word in seq:
            word_freq[word] += 1
        label_set.add(label)
    temp = sorted(word_freq.items(), key=lambda x: x[1], reverse=False)
    words, _ = list(zip(*temp))
    word_idx = dict(list(zip(words, range(len(words)))))
    word_idx['<unk>'] = len(words)
    word_idx['<pad>'] = len(words) + 1
    label_idx = dict(list(zip(label_set, range(len(label_set)))))
    return word_idx, label_idx


train_data = load_data('model/Text Classification/data/new.txt', False)
word_idx, label_idx = build_dict(train_data)

vocab_size = len(word_idx) + 1
epochs = 2


class MyLSTM(paddle.nn.Layer):
    def __init__(self):
        super(MyLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, 256)
        self.lstm = nn.LSTM(256, 128, num_layers=2, direction='bidirect', dropout=0.5)
        self.linear = nn.Linear(in_features=128 * 2, out_features=len(label_idx))
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        emb = self.dropout(self.embedding(inputs))
        output, (h, c) = self.lstm(emb)
        x = paddle.mean(output, axis=1)
        # x形状大小为[batch_size, hidden_size * num_directions]
        x = self.dropout(x)
        return self.linear(x)



import numpy as np


def text_to_input(text, word_idx):
    words = text.split()
    seq = [word_idx.get(word, word_idx['<unk>']) for word in words]
    return np.array(seq)

# # 创建模型实例
# model = MyLSTM()
#
# # 加载已训练好的权重
# model_state_dict = paddle.load('model/Text Classification/result/model_final1.pdparams')
# model.set_state_dict(model_state_dict)
#
# test_text = "美轮美奂的传统建筑，古典花园里透出的西班牙风情，让你会情不自禁地爱上那里。"
# # 将文本转换为模型输入格式
# input_seq = text_to_input(test_text, word_idx)
#
# # 转换为 paddle.Tensor
# input_tensor = paddle.to_tensor(input_seq, dtype='int32').unsqueeze(0)
#
# # 使用模型进行预测
# model.eval()
# with paddle.no_grad():
#     output = model(input_tensor)
#     # 假设输出为概率分布，可以使用 argmax 获取预测结果
#     predicted_label_idx = paddle.argmax(output, axis=1).item()
#
# inverted_dict = {v: k for k, v in label_idx.items()}
# # 获取预测标签
# predicted_label = inverted_dict[predicted_label_idx]
# # print(inverted_dict)
# print("Predicted Label:", predicted_label)
