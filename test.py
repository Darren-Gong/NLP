import jieba
import os
import torch
import paddle
from transformers import BertTokenizer
from function2 import BertNer, predict_sentence
device = "cuda" if torch.cuda.is_available() else "cpu"
bert_path = r"model/NER/bert-base-chinese"
def text_processing_model2(text):
    # 模型2的文本处理逻辑
    model = BertNer().to(device)
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    model.load_state_dict(torch.load( "model/NER/model.pth"))
    original_sentence, entities = predict_sentence(model, tokenizer, text)
    print("原句:", original_sentence)
    print("命名实体:")
    for entity, label in entities:
        print(f"实体: {entity}, 类别: {label}")

    return "模型2处理结果：" + text.capitalize()

def process_file(selected_model, file_content):

    if selected_model == 'model2':
        processed_text = text_processing_model2(file_content)
    else:
        processed_text = "未选择有效的模型"

    return processed_text
with open('uploads/文本2.txt', 'r', encoding='utf-8') as f:
    file_content = f.read()
    processed_text = process_file('model2', file_content)
    print('result:',processed_text)