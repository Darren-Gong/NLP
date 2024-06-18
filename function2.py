

import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertModel
"""
以下为命名实体识别部分

"""
bert_path = r"model/NER/bert-base-chinese"
device = "cuda" if torch.cuda.is_available() else "cpu"
###模型###
ner_classes_list = ["O", "B_Abstract", "I_Abstract", "B_Location", "I_Location", "B_Metric", "I_Metric", "B_Thing",
                    "I_Thing", "B_Time", "I_Time", "B_Organization", "I_Organization", "B_Physical", "I_Physical",
                    "B_Person", "I_Person", "X", "[CLS]", "[SEP]", "[PAD]"]

class BertNer(nn.Module):
    def __init__(self):
        super(BertNer, self).__init__()
        self.dropout = nn.Dropout()
        self.bert = BertModel.from_pretrained(bert_path)
        self.bilstm = nn.LSTM(bidirectional=True, num_layers=1, input_size=768,
                              hidden_size=768 // 2, batch_first=True)
        self.crf = CRF(num_tags=len(ner_classes_list), batch_first=True)
        self.classifier = nn.Linear(768, len(ner_classes_list))

    def forward(self, bert_input, batch_labels: None = None):
        attention_mask = bert_input['attention_mask']

        bert_output = self.bert(**bert_input).last_hidden_state
        lstm_output, _ = self.bilstm(bert_output)
        logits = self.classifier(lstm_output)
        outputs = (logits,)
        if batch_labels is not None:
            loss = self.crf(emissions=logits, tags=batch_labels, mask=attention_mask.gt(0)) * -1
            outputs = (loss,) + outputs
        return outputs

def predict_sentence(model, tokenizer, sentence):
    inputs = tokenizer(sentence, max_length=512, padding=True, truncation=True, return_tensors="pt").to(device)
    logits = model(bert_input=inputs)[0]
    batch_output = model.crf.decode(logits, mask=inputs['attention_mask'].gt(0))

    # 获取预测的标签序列
    pred_labels = batch_output[0]

    # 解码预测的命名实体
    entities = []
    current_entity = ''
    current_label = ''
    for i, label_id in enumerate(pred_labels):
        label = ner_classes_list[label_id]
        if label.startswith('B_'):
            if current_entity:
                entities.append((current_entity, current_label))
            current_entity = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][i].item())
            current_label = label[2:]
        elif label.startswith('I_'):
            current_entity += tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][i].item())
        elif label == 'O' and current_entity:
            entities.append((current_entity, current_label))
            current_entity = ''
            current_label = ''

    if current_entity:
        entities.append((current_entity, current_label))

    return sentence, entities

