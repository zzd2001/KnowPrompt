import argparse
import os
import numpy as np
import pandas as pd
from pandas import DataFrame
import json
from collections import Counter
import logging

logger = logging.getLogger(__name__)

def get_labels(path, name, negative_label="no_relation"):
    """See base class."""
    """
        这个函数的作用：
        这个函数将打开数据集文件目录下的train.txt文件，这个文件用来训练模型
        文件中每一行都是用字符串表示的一个字典，字典中4个键值对，分别是token、头实体、尾实体、关系
        将这些字符串表示的字典转成python数据类型字典，然后存到列表feature中
        然后返回feature列表
    """
    
    count = Counter()
    with open(os.path.join(path, name), "r", encoding="utf-8") as f:
        features = []
        for line in f.readlines():
            line = line.rstrip()
            if len(line) > 0:
                features.append(eval(line))
    return features

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=16,
        help="Training examples for each class.")
    parser.add_argument("--seed", type=int, nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Random seeds")
    parser.add_argument("--data_dir", type=str, default="../datasets/", help="Path to original data")
    parser.add_argument("--dataset", type=str, default="tacred", help="Path to original data")
    parser.add_argument("--data_file", type=str, default='train.txt', choices=['train.txt', 'val.txt'], help="k-shot or k-shot-10x (10x dev set)")
    parser.add_argument("--mode", type=str, default='k-shot', choices=['k-shot', 'k-shot-10x'], help="k-shot or k-shot-10x (10x dev set)")

    args = parser.parse_args()

    path = os.path.join(args.data_dir, args.dataset)
    output_dir = os.path.join(path, args.mode)
    dataset = get_labels(path, args.data_file)

    for seed in args.seed:
        np.random.seed(seed)
        np.random.shuffle(dataset)

        k = args.k
        setting_dir = os.path.join(output_dir, f"{k}-{seed}")
        os.makedirs(setting_dir, exist_ok=True)

        label_list = {}
        for line in dataset:
            label = line['relation']
            if label not in label_list:
                label_list[label] = [line]
            else:
                label_list[label].append(line)

        with open(os.path.join(setting_dir, "train.txt"), "w", encoding="utf-8") as f:
            for label in label_list:
                for line in label_list[label][:k]:
                    f.writelines(json.dumps(line, ensure_ascii=False))
                    f.write('\n')

if __name__ == "__main__":
    main()
