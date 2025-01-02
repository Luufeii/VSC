### Extract the fetures of all category, then generate one feature matrix for each class (one json file)


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
import argparse
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 容忍损坏图片


def loader(path):
    return Image.open(path)


def transform(path):
    trans = transforms.Compose(
        [transforms.ToTensor()])  # 这里不进行归一化
    img=loader(path)
    img=img.resize((224, 224))
    img=trans(img)
    return img


if __name__ == '__main__':

    parser=argparse.ArgumentParser()
    # 这里要给出的数据路径是类别名的前一级目录为止。
    parser.add_argument('--data_dir',type=str,default='',help='root of dataset')
    parser.add_argument('--pretrain_model',type=str,default='',help='path of pretrained model to extract features')
    # parser.add_argument('--mode',type=str,default='SS',help='SS|PS')
    parser.add_argument('--gpu_id',type=int,default=0, help='Single GPU')
    opts=parser.parse_args()

    #CUB_fn = os.path.join(root, "zsl_data", "CUB_200_2011", "images")
    #AwA2_fn=os.path.join(root,"zsl_data","Animals_with_Attributes2","JPEGImages")
    #SUN_fn=os.path.join(root,"zsl_data","SUN","image")
    #AwA1_fn="/home/ziyu/zsl_data/AwA/images"

    device=torch.device(opts.gpu_id)

    ResNet = model.resnet18(weights=model.ResNet18_Weights.IMAGENET1K_V1)
    ResNet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    x = ResNet.fc.in_features
    if opts.pretrain_model!="":
        ResNet.fc = nn.Linear(x, 9)
        ResNet.load_state_dict(torch.load(opts.pretrain_model)) ##"/home/ziyu/DNN_RC/Tools/CUB_FT_SS_101_22.pkl"
    ResNet = nn.Sequential(*list(ResNet.children())[:-1])  # 去掉了全连接层
    ResNet = ResNet.to(device)
    ResNet.eval()

    data_fn = opts.data_dir

    fea = []
    label = []
    # os.listdir用于列出指定目录下的所有文件和子目录名称。
    for x in os.listdir(data_fn):  # x是该目录下的类别名
        cur = os.path.join(data_fn, x)
        cur_class_list = os.listdir(cur)  # 这是该类别下的每个图片的文件名组成的列表

        all_features=[]
        for image_id in cur_class_list:  # Load each image of each class

            image_fn = os.path.join(cur, image_id)
            image = transform(image_fn)  # 加载图片
            image = image.unsqueeze(0)  # 4 dimensions input，增加了一个批次维度，模型的输入是(B,C,H,W)四个维度
            image = image.to(device)
            features = ResNet(image)
            f = features.to("cpu")
            f = f.detach().numpy()  # 脱离梯度计算图，然后转换为数组
            fea_vec = f[0].reshape(-1)  # 将特征展为一维数组
            # print(fea_vec.shape)
            fea_vec = torch.tensor(fea_vec)
            fea_vec = F.normalize(fea_vec, dim=0)
            fea_vec = fea_vec.numpy()
            all_features.append(fea_vec.tolist())

        obj = {"features": all_features}
        cur_url = os.path.join(cur, "ResNet18_feature.json")
        print("%s has finished ..."%(x))
