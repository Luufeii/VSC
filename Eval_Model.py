### Test the model performance

import os
import torch
import torchvision.models as model
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
from numpy import linspace
from matplotlib import cm
import json
import matplotlib.pyplot as plt
import argparse

from PIL import Image


def L2_dis(x, y):
    return np.sum((x - y) * (x - y))


def NN_search(x, center):
    ret = ""
    MINI = -1
    for c in center.keys():
        tmp = L2_dis(x, center[c])
        if MINI == -1:
            MINI = tmp
            ret = c
        if tmp < MINI:
            MINI = tmp
            ret = c
    return ret


def get_center(checkpoint_fn):
    center = {}
    file = "Pred_Center.txt"
    center_fn = os.path.join(checkpoint_fn, file)
    with open(center_fn, "r") as f:
        for i, lines in enumerate(f):
            line = lines.strip().split()
            pp = [float(x) for x in line]
            center[target_class[i]] = np.array(pp)

    return center


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Testing cZSL parameters setting")
    # parser.add_argument('--data_path',type=str,default="/home/ziyu/zsl_data")
    parser.add_argument('--GPU', type=int, default=0)
    # parser.add_argument('--dataset', type=str, default="")
    # parser.add_argument('--split_mode', type=str, default="standard_split")
    parser.add_argument('--checkpoint_fn', type=str, default="trained_models")
    parser.add_argument('--root', type=str, default="wafer_data")
    opts = parser.parse_args()

    args = vars(opts)  # 返回opts的__dict__属性
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))

    device = torch.device(opts.GPU)

    target_class = []
    # test_class_fn = os.path.join(root, "zsl_data", "proposed_split", "CUB", "testclasses.txt")
    # test_class_fn = os.path.join(root, "zsl_data", "proposed_split", "AWA2", "testclasses.txt")
    # test_class_fn = os.path.join(root, "zsl_data", "standard_split", "AWA2", "testclasses.txt")
    # test_class_fn = os.path.join(root, "zsl_data", "standard_split", "SUN", "testclasses.txt")
    # test_class_fn = os.path.join(root, "zsl_data", "standard_split", "AWA1", "testclasses.txt")
    # test_class_fn = "/home/ziyu/zsl_data/SUN/SUN10_test.txt"
    # test_class_fn = os.path.join(root, "zsl_data", opts.split_mode, opts.dataset.upper(), "testclasses.txt")
    '''
    if opts.dataset == "SUN10":
        test_class_fn = os.path.join(opts.root, 'zsl_data', 'SUN', 'SUN10_test.txt')
    else:
        test_class_fn = os.path.join(opts.root, "zsl_data", opts.split_mode, opts.dataset.upper(), "testclasses.txt")
    '''
    test_class_fn = os.path.join(opts.root, 'test_classes.txt')

    with open(test_class_fn, "r") as f:
        for lines in f:
            line = lines.strip().split()
            target_class.append(line[0])
            # Get the name of unseen classes
    '''
    if opts.split_mode == 'standard_split':
        mode = "SS"
    else:
        mode = "PS"
    '''
    All = 0.0
    '''
    if opts.dataset == "CUB":
        img_fn = os.path.join(opts.root, "zsl_data", "CUB_200_2011", "images")
    if opts.dataset == "AwA2":
        img_fn = os.path.join(opts.root, "zsl_data", "Animals_with_Attributes2", "JPEGImages")
    if opts.dataset == "AwA1":
        img_fn = os.path.join(opts.root, "zsl_data", "AwA", "images")
    if opts.dataset == "SUN" or opts.dataset == "SUN10":
        img_fn = "/home/ziyu/zsl_data/SUN/image"
    '''
    img_fn = os.path.join(opts.root, "imgs", "test")

    # CUB_fn=os.path.join(root,"zsl_data","CUB_200_2011", "images")
    # AWA2_fn=os.path.join(root,"zsl_data","Animals_with_Attributes2", "JPEGImages")
    # AWA1_fn = os.path.join(root, "zsl_data", "AwA", "images")
    # SUN_fn="/home/ziyu/zsl_data/SUN/image"
    center = get_center(opts.checkpoint_fn)

    # center=RC_2_SC(i)

    total_correct = 0
    total_sum = 0
    for id, target in enumerate(target_class):
        cur = os.path.join(img_fn, target)

        fea_name = ""

        url = os.path.join(cur, "ResNet18_feature.json")  # 这是前面已经提取好了的视觉特征，保存在每个类别名文件夹下
        js = json.load(open(url, "r"))
        cur_features = js["features"]

        correct = 0
        sum = 0

        for fea_vec in cur_features:  #### Test the image features of each class
            fea_vec = np.array(fea_vec)
            ans = NN_search(fea_vec, center)  # Find the nearest neighbour in the feature space

            if ans == target:
                total_correct += 1
                correct += 1
            sum += 1
            total_sum += 1

        All += correct * 1.0 / sum

    # assert test_class_num==len(target_class), "Maybe there is something wrong?"
    print("The final MCA result is %.5f" % (All / len(target_class)))  # MCA是每个类的准确率相加求均值
    print(f'所有样本的准确率是{total_correct/total_sum}')
