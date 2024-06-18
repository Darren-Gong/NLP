import jieba
import numpy as np
import base64
import torch
import paddle
from transformers import BertTokenizer
import matplotlib.pyplot as plt
from function1 import load_data, build_dict, text_to_input, MyLSTM
from function2 import BertNer, predict_sentence
from function4 import loading_source, plot_result,plot_result1
import pandas as pd
import time
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.cluster import KMeans,Birch, DBSCAN
from sklearn.decomposition import PCA
device = "cuda" if torch.cuda.is_available() else "cpu"
bert_path = r"model/NER/bert-base-chinese"
# 这里面是后端调用代码的集合

# 分词
def text_processing_model1(text):
    # 模型1的文本处理逻辑
    words = jieba.cut(text)

    return '/ '.join(words)


def text_processing_model2(text):
    # 模型2的文本处理逻辑
    model = BertNer().to(device)
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    model.load_state_dict(torch.load( "model/NER/model.pth"))
    original_sentence, entities = predict_sentence(model, tokenizer, text)
    result = ""
    result += "原句: " + original_sentence + "<br>"
    result += "命名实体:<br>"
    for entity, label in entities:
        result += f"实体: {entity}, 类别: {label}<br>"

    return "命名实体识别结果：" + result


def text_processing_model3(test_text):
    # 加载训练数据和构建字典
    train_data = load_data('model/Text Classification/data/new.txt', False)
    word_idx, label_idx = build_dict(train_data)

    # 创建模型实例并加载预训练权重
    model = MyLSTM()
    model_state_dict = paddle.load('model/Text Classification/result/model_final1.pdparams')
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
    return "的文本分类结果：" + predicted_label


def text_processing_model4(label_path, ori_path):
    result = {}

    sentences = loading_source(file_name=ori_path)
    # 词频矩阵
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.46)
    transformer = TfidfTransformer()
    # Fit Raw Documents
    freq_words_matrix = vectorizer.fit_transform(sentences)
    # Get Words Of Bag

    words = vectorizer.get_feature_names_out()
    tfidf = transformer.fit_transform(freq_words_matrix)
    weight = freq_words_matrix.toarray()
    # 降维
    pca = PCA(n_components=10)
    training_data = pca.fit_transform(weight)
    # 真实标签用于计算指标
    content = pd.read_csv(label_path)
    labels_true = content.flag.to_list()
    # 最优num=4，进行kmeans聚类
    num_of_class = 4
    clf = KMeans(n_clusters=num_of_class, max_iter=10000, init="k-means++", tol=1e-6)
    result1 = clf.fit(training_data)
    source = list(clf.predict(training_data))
    label = clf.labels_
    # 生成指标
    ars = metrics.adjusted_rand_score(labels_true, label)
    result['adjusted_rand_score'] = ars

    fmi = metrics.fowlkes_mallows_score(labels_true, label)
    result['FMI'] = fmi

    silhouette = metrics.silhouette_score(training_data, label)
    result['silhouette'] = silhouette

    CHI = metrics.calinski_harabasz_score(training_data, label)
    result['CHI'] = CHI

    # 可视化结果
    plot_result(training_data, source, num_of_class)
    # 将图片转换为 Base64 编码字符串
    with open("plot/plot.png", "rb") as img_file:
        plot_base64 = base64.b64encode(img_file.read()).decode('utf-8')
    # 构建结果字符串，包括三个指标和图片路径
    result_str = f"文本聚类结果：<br>ARS: {ars}<br>FMI: {fmi}<br>Silhouette: {silhouette}<br>CHI: {CHI}"

    return result_str, plot_base64


def text_processing_model5(label_path, ori_path):
    result = {}

    sentences = loading_source(file_name=ori_path)
    # 词频矩阵
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.46)
    transformer = TfidfTransformer()
    # Fit Raw Documents
    freq_words_matrix = vectorizer.fit_transform(sentences)
    # Get Words Of Bag
    weight = freq_words_matrix.toarray()
    # 降维
    pca = PCA(n_components=10)
    training_data = pca.fit_transform(weight)
    # 真实标签用于计算指标
    content = pd.read_csv(label_path)
    labels_true = content.flag.to_list()
    # 最优num=4，进行kmeans聚类
    numOfClass1: int = 4

    start1 = time.time()
    clf1 = Birch(n_clusters=4, branching_factor=10, threshold=0.01)

    result1 = clf1.fit(training_data)
    source1 = list(clf1.predict(training_data))
    end1 = time.time()

    label1 = clf1.labels_
    # 生成指标
    ars1 = metrics.adjusted_rand_score(labels_true, label1)
    result['adjusted_rand_score'] = ars1

    fmi1 = metrics.fowlkes_mallows_score(labels_true, label1)
    result['FMI'] = fmi1

    silhouette1 = metrics.silhouette_score(training_data, label1)
    result['silhouette'] = silhouette1

    CHI1 = metrics.calinski_harabasz_score(training_data, label1)
    result['CHI'] = CHI1

    # 可视化结果
    plot_result1(training_data, source1, numOfClass1)
    # 将图片转换为 Base64 编码字符串
    with open("plot/plot1.png", "rb") as img_file:
        plot_base64 = base64.b64encode(img_file.read()).decode('utf-8')
    # 构建结果字符串，包括三个指标和图片路径
    result_str = f"文本聚类结果：<br>ARS: {ars1}<br>FMI: {fmi1}<br>Silhouette: {silhouette1}<br>CHI: {CHI1}"

    return result_str, plot_base64
def text_processing_model6(label_path, ori_path):
    result = {}

    sentences = loading_source(file_name=ori_path)
    # 词频矩阵
    vertorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.46)
    transformer = TfidfTransformer()
    # Fit Raw Documents
    freq_words_matrix = vertorizer.fit_transform(sentences)
    # Get Words Of Bag
    words = vertorizer.get_feature_names_out()
    tfidf = transformer.fit_transform(freq_words_matrix)
    weight = freq_words_matrix.toarray()
    content = pd.read_csv(label_path)
    labels_true = content.flag.to_list()
    pca = PCA(n_components=10)
    training_data = pca.fit_transform(weight)
    pca1 = PCA(n_components=8)
    trainingData1 = pca1.fit_transform(weight)
    # 真实标签用于计算指标

    ##DBSCAN
    db = DBSCAN(eps=0.08, min_samples=7)
    result2 = db.fit(trainingData1)
    source2 = list(db.fit_predict(trainingData1))
    label2 = db.labels_
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    # 生成指标
    ars1 = metrics.adjusted_rand_score(labels_true, label2)
    result['adjusted_rand_score'] = ars1

    fmi1 = metrics.fowlkes_mallows_score(labels_true, label2)
    result['FMI'] = fmi1

    silhouette1 = metrics.silhouette_score(trainingData1, label2)
    result['silhouette'] = silhouette1

    CHI1 = metrics.calinski_harabasz_score(trainingData1, label2)
    result['CHI'] = CHI1

    colors = plt.cm.Spectral(np.linspace(0, 1, len(set(label2))))
    for k, col in zip(set(label2), colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
        class_member_mask = (label2 == k)
        xy = training_data[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=10)
        xy = training_data[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)
    plt.savefig('plot/plot2.png')
    # 将图片转换为 Base64 编码字符串
    with open("plot/plot2.png", "rb") as img_file:
        plot_base64 = base64.b64encode(img_file.read()).decode('utf-8')
    # 构建结果字符串，包括三个指标和图片路径
    result_str = f"文本聚类结果：<br>ARS: {ars1}<br>FMI: {fmi1}<br>Silhouette: {silhouette1}<br>CHI: {CHI1}"

    return result_str, plot_base64

