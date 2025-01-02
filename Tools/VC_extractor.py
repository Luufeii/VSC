### For each dataset, find the feature matrix of each class, then calculate the mean (Visual Center), finally normalize.

import torch
import os
import torchvision.models as model
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
from PIL import Image
import torch.nn.functional as F
import json
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # train_class_list是一个文本的地址，文本里的内容每一行的第一个字符是训练标签
    parser.add_argument('--train_class_list', type=str, default='', help='Train(Seen) class list location')
    parser.add_argument('--test_class_list', type=str, default='', help='Test(Unseen) class list location')
    parser.add_argument('--data_dir', type=str, default='', help='root of corresponding dataset')
    parser.add_argument('--feature_name', type=str, default='ResNet18_feature.json')
    # parser.add_argument('--dataset_name',type=str,default='CUB')
    # parser.add_argument('--mode',type=str,default='SS',help='SS|PS')
    opts = parser.parse_args()

    target_class = []
    test_class = []
    # target_class_fn = os.path.join(root, "zsl_data", "proposed_split", "CUB", "trainvalclasses.txt")
    # test_class_fn=os.path.join(root, "zsl_data", "proposed_split", "CUB", "testclasses.txt")
    # target_class_fn=os.path.join(root,"zsl_data","proposed_split","AWA2","trainvalclasses.txt")
    # test_class_fn=os.path.join(root,"zsl_data","proposed_split","AWA2","testclasses.txt")
    # target_class_fn=os.path.join(root,"zsl_data","standard_split","AWA2","trainvalclasses.txt")
    # test_class_fn=os.path.join(root,"zsl_data","standard_split","AWA2","testclasses.txt")
    # target_class_fn=os.path.join(root,"zsl_data","standard_split","AWA1","trainvalclasses.txt")
    # test_class_fn=os.path.join(root,"zsl_data","standard_split","AWA1","testclasses.txt")
    # target_class_fn=os.path.join(root,"zsl_data","standard_split","SUN","trainvalclasses.txt")
    # test_class_fn=os.path.join(root,"zsl_data","standard_split","SUN","testclasses.txt")
    # target_class_fn="/home/ziyu/zsl_data/SUN/SUN10_train.txt"
    # test_class_fn="/home/ziyu/zsl_data/SUN/SUN10_test.txt"
    target_class_fn = os.path.join(opts.train_class_list)
    test_class_fn = os.path.join(opts.test_class_list)

    with open(target_class_fn, "r") as f:
        for lines in f:
            # strip()方法‌：用于移除字符串头尾的指定字符（默认为空格或换行符）
            # split()方法‌：通过指定的分隔符对字符串进行切片。默认情况下，split()使用任何空白字符作为分隔符，包括空格、制表符、换行符等。
            line = lines.strip().split()
            target_class.append(line[0])
            # seen class

    with open(test_class_fn, "r") as f:
        for lines in f:
            line = lines.strip().split()
            test_class.append(line[0])
            # unseen class, even if this is useless

    target_VC = []
    for x in target_class:
        url = os.path.join(opts.data_dir, x, opts.feature_name)
        f = json.load(open(url, "r"))
        features = f['features']
        sum = [0.0] * 512  # resnet18提取的特征是512维的
        sum = np.array(sum)
        cnt = 0  # 计数
        for y in features:
            cnt += 1
            sum += y

        sum /= cnt
        avg = torch.tensor(sum)
        avg = F.normalize(avg, dim=0)  # 取归一化的平均值作为视觉中心
        target_VC.append(avg.numpy().tolist())

    test_VC = []
    for x in test_class:
        url = os.path.join(opts.data_dir, x, opts.feature_name)
        f = json.load(open(url, "r"))
        features = f['features']
        sum = [0.0] * 512
        sum = np.array(sum)
        cnt = 0
        for y in features:
            cnt += 1
            sum += y

        sum /= cnt
        avg = torch.tensor(sum)
        avg = F.normalize(avg, dim=0)
        test_VC.append(avg.numpy().tolist())

    obj = {"train": target_VC, "test": test_VC}
    cur_url = "../json_file/ResNet18_VC.json"
    json.dump(obj, open(cur_url, "w"))
