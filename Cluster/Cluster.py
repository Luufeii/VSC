### Use K-means to calculate the approximated visuals center of unseen classes

import json
import os
import torch
from torch.utils.data import DataLoader
import torchvision.models as model
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering  # 谱聚类，是一种聚类方法
import torch.nn.functional as F
import numpy as np
import argparse


def Cluster(features, opts):

    if opts.cluster_method == 'Kmeans':
        KM(features, opts)
    else:
        SC(features, opts)


def KM(features, opts):
    clf = KMeans(n_clusters=opts.center_num, n_init=50, max_iter=100000, init="k-means++")

    print("Start Cluster ...")
    s = clf.fit(features)
    print("Finish Cluster ...")

    obj = {}
    obj["VC"] = clf.cluster_centers_.tolist()

    print('Start writing ...')
    saved_url = "../json_file/Cluster_VC_ResNet.json"
    if not os.path.exists(saved_url):
        os.makedirs(saved_url)
    json.dump(obj, open(saved_url, "w"))
    print("Finish writing ...")


def SC(features, opts):
    Spectral = SpectralClustering(n_clusters=opts.center_num, eigen_solver='arpack', affinity="nearest_neighbors")

    print("Start Cluster ...")
    pred_class = Spectral.fit_predict(features)
    print("Finish Cluster ...")

    belong = Spectral.labels_
    sum = {}
    count = {}
    for i, x in enumerate(features):
        label = belong[i]
        if sum.get(label) is None:
            sum[label] = [0.0] * 2048
            count[label] = 0
        for j, y in enumerate(x):
            sum[label][j] += y
        count[label] += 1

    all_cluster_center = []
    for label in sum.keys():

        for i, x in enumerate(sum[label]):
            sum[label][i] /= (count[label] * 1.0)

        all_cluster_center.append(sum[label])

    print("Start writing ...")
    obj = {}
    url = "../json_file/Cluster_VC_ResNet.json"
    obj["VC"] = all_cluster_center
    json.dump(obj, open(url, "w"))
    print("Finish writing ...")


if __name__ == '__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('--test_class_list',type=str,default='',help="Test(Unseen) class list location")
    # parser.add_argument('--mode',type=str,default='SS',help='Split method')
    parser.add_argument('--data_dir',type=str,default='',help='root of corresponding dataset')
    parser.add_argument('--feature_name',type=str,default='ResNet18_feature.json')
    parser.add_argument('--cluster_method',type=str,default='Kmeans',help='choose the cluster algorithm')
    parser.add_argument('--center_num',type=int,default=29,help='unseen class num')
    # parser.add_argument('--dataset_name', type=str, default='CUB')
    opts = parser.parse_args()

    test_class_fn = opts.test_class_list

    test_class = []
    with open(test_class_fn, "r") as f:
        for lines in f:
            line = lines.strip().split()
            test_class.append(line[0])

    all_features = []
    for x in test_class:
        url = os.path.join(opts.data_dir, x, opts.feature_name)
        f = json.load(open(url, "r"))
        for y in f['features']:
            all_features.append(y)

    Cluster(all_features, opts)
