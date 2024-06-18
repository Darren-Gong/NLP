# -*- coding: utf-8 -*-
import pandas as pd
import jieba
import csv
import matplotlib.pyplot as plt
import numpy as np
"""
以键入文本路径为输入，通过键入待处理文件和输出文件，
完成分词，便于后续连接前端。
测试例为，input.csv
        out.csv
连接前端可以直接自动命名输入文件为input.csv，输出为out.csv,省去输入
"""
class WordCut(object):
    def __init__(self, stopwords_path="model/stopwords.txt"):
        self.stopwords_path = stopwords_path

    def addDictionary(self, dict_list):
        map(lambda x: jieba.load_userdict(x), dict_list)

    def seg_sentence(self, sentence, stopwords_path=None):
        if stopwords_path is None:
            stopwords_path = self.stopwords_path

        def stopwordslist(filepath):
            stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
            return stopwords

        sentence_seged = jieba.cut(sentence.strip())
        stopwords = stopwordslist(stopwords_path)
        outstr = ''
        for word in sentence_seged:
            if word not in stopwords:
                if word != '\t':
                    outstr += word
                    outstr += " "
        return outstr

def process_csv(input_file, output_file):
    word_cutter = WordCut()
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        writer.writerow(["index", "content"])
        next(reader)

        for index, row in enumerate(reader):
            if len(row) >= 3:
                seg_result = word_cutter.seg_sentence(row[2])
                writer.writerow([index, seg_result])

def loading_source(file_name: str)->list:
    source_df = pd.read_csv(file_name, sep=',', encoding='utf-8')
    source_df.dropna(inplace=True)
    return source_df.content.values.tolist()
def plot_result(data, cluster_res, cluster_num, algorithm='None'):
    nPoints = len(data)
    scatter_colors = ['#ac4a5a','#a5903c', '#6d49ef', '#59c280']
    for i in range(cluster_num):
        color = scatter_colors[i % len(scatter_colors)]
        x1 = []
        y1 = []
        for j in range(nPoints):
            if cluster_res[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])
        plt.scatter(x1, y1, c=color, alpha=1, marker='o')
        plt.plot(marksize=10)
        plt.savefig('plot/plot.png')
    plt.show()
def plot_result1(data, cluster_res, cluster_num, algorithm='None'):
    nPoints = len(data)
    scatter_colors = ['#ac4a5a','#a5903c', '#6d49ef', '#59c280']
    for i in range(cluster_num):
        color = scatter_colors[i % len(scatter_colors)]
        x1 = []
        y1 = []
        for j in range(nPoints):
            if cluster_res[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])
        plt.scatter(x1, y1, c=color, alpha=1, marker='o')
        plt.plot(marksize=10)
        plt.savefig('plot/plot1.png')
    plt.show()


