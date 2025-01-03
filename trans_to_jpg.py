# 这里把wm38中的数据保存为单通道的png格式，jpg格式会进行有损压缩，导致像素失真
# 其中，像素0、1、2分别转换为0、128、255，如果效果太好的话就直接按照0、1、2像素保存

# 每个类别文件夹下要生成文件"ResNet18_feature.json"

import numpy as np
import os
from PIL import Image

# 加载数据
train_data = np.load('wafer_data/data_npz/train_WM.npz')
val_data = np.load('wafer_data/data_npz/val_WM.npz')
test_data = np.load('wafer_data/data_npz/test_WM.npz')

train_wm = train_data['denoise_wm']
train_label = train_data['label_name']

val_wm = val_data['denoise_wm']
val_label = val_data['label_name']

test_wm = test_data['denoise_wm']
test_label = test_data['label_name']

# WM = train_wm[1]
# print(type(WM))
# print(WM.shape)

# 创建文件夹
unique_label = np.unique(val_label)
for dir1 in ['train', 'val', 'test']:
    for dir2 in unique_label:
        create_dir = f'wafer_data/imgs/{dir1}/{dir2}'
        try:
            os.makedirs(create_dir)
        except FileExistsError:
            print(f'文件夹{create_dir}已存在')


# 转换像素的函数
def trans_pixel(array):
    tmp = array
    H = array.shape[0]
    W = array.shape[1]
    for i in range(H):
        for j in range(W):
            if int(tmp[i][j]) == 1:
                tmp[i][j] = 128
            elif int(tmp[i][j]) == 2:
                tmp[i][j] = 255

    return tmp


# 开始转换图片
for i in range(len(train_wm)):
    wm = train_wm[i]
    wm = trans_pixel(wm)
    label = train_label[i]
    wm = wm.astype(np.uint8)
    # wm = wm.reshape(52,52,1) # 一个通道的话，就不用加上通道维度了，否则会报错
    save_path = f'wafer_data/imgs/train/{label}/{i+1}.png'
    image = Image.fromarray(wm)
    image.save(save_path, 'PNG')

for i in range(len(val_wm)):
    wm = val_wm[i]
    wm = trans_pixel(wm)
    label = val_label[i]
    wm = wm.astype(np.uint8)
    # wm = wm.reshape(52,52,1) # 一个通道的话，就不用加上通道维度了，否则会报错
    save_path = f'wafer_data/imgs/val/{label}/{i+1}.png'
    image = Image.fromarray(wm)
    image.save(save_path, 'PNG')

for i in range(len(test_wm)):
    wm = test_wm[i]
    wm = trans_pixel(wm)
    label = test_label[i]
    wm = wm.astype(np.uint8)
    # wm = wm.reshape(52,52,1) # 一个通道的话，就不用加上通道维度了，否则会报错
    save_path = f'wafer_data/imgs/test/{label}/{i+1}.png'
    image = Image.fromarray(wm)
    image.save(save_path, 'PNG')
