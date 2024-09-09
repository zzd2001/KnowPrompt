from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
import re
import torch
import json

# 中文分词模型
# model_name_or_path = "roberta-large"
model_name_or_path = "bert-base-chinese"

dataset_name = "historyRE"

model_path = '/root/KnowPrompt/bert-base-chinese'


tokenizer = AutoTokenizer.from_pretrained(model_path)
def split_label_words(tokenizer, label_list):
    """
        这个函数的作用：
        每一个数据集中都有一个“rel2id.json”文件，文件里是一个字典，键值对由本数据集中用到的“关系”及其对应的id序号组成
        把这个字典转成列表，就得到参数label_list的实参
        把label_list中的“关系”取出来，使用模型的分词器对其进行分词、编码，用一个列表存放编码结果
        再把这个列表转成一维张量，label_word_list中就是一个个这样的张量
        然后将label_word_list进行填充，达到格式上的一致，然后转成二维张量，就得到本函数的输出结果
    """
    label_word_list = []
    for label in label_list:
        if label == 'no_relation' or label == "NA":
            label_word_id = tokenizer.encode('no relation', add_special_tokens=False)
            label_word_list.append(torch.tensor(label_word_id))
        else:
            tmps = label
            label = label.lower()
            label = label.split("(")[0]
            label = label.replace(":"," ").replace("_"," ").replace("per","person").replace("org","organization")
            label_word_id = tokenizer(label, add_special_tokens=False)['input_ids']
            print(label, label_word_id)
            label_word_list.append(torch.tensor(label_word_id))
    padded_label_word_list = pad_sequence([x for x in label_word_list], batch_first=True, padding_value=0)
    return padded_label_word_list

with open(f"dataset/{dataset_name}/rel2id.json", "r") as file:
    t = json.load(file)
    label_list = list(t)

t = split_label_words(tokenizer, label_list)

with open(f"./dataset/{model_name_or_path}_{dataset_name}.pt", "wb") as file:
    torch.save(t, file)