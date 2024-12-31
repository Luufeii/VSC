### Use training set to finetune the ImageNet pretrain model. Mainly for fine-grained dataset

import torch
import torchvision.models as models
import torch.nn as nn
import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
import torch.optim as optim
import time
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.optim import lr_scheduler
import copy
import argparse


def default_loader(path):
    # return Image.open(path).convert('RGB') # 这里convert('RGB')是为了消除原始图像RGBA中的A通道(透明通道)
    # 我们这里将晶圆图保存为了灰度图像jpg，只有单通道
    return Image.open(path)


def get_ResNet(device):
    # ResNet=models.resnet101(pretrained=True)
    ResNet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # 为了对比，这里也使用res18
    # 将第一个卷积的输入通道改为1
    ResNet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    x=ResNet.fc.in_features
    print(x)
    print("****")
    # ResNet.fc=nn.Linear(x,150)
    ResNet.fc = nn.Linear(x, 9)  # 我们只使用9种单故障进行微调
    ResNet=ResNet.to(device)
    return ResNet


class ImageFolder(Dataset):  # 这个类实例化之后就是一个Dataset对象
    def __init__(self, imgs, transform=None, loader=default_loader):

        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        idx, fn = self.imgs[index]
        img = self.loader(fn)
        # img=img.resize((224,224))
        if self.transform is not None:
            img = self.transform(img)
        return img, idx  # idx是样本对于类别的标号

    def __len__(self):
        return len(self.imgs)


def get_data_loader(opts):
    # 由于数据集特性，我们不对数据进行裁剪和归一化
    data_transform = {
        'train': transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(224),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    ys = {}  # 字典，key是可见类(训练类)的标签，value是标签在opts.train_class_list中的位置
    pos = 0
    train_class = []
    # opts.train_class_list是一个文本的地址，文本里的内容每一行的第一个字符是训练标签
    with open(opts.train_class_list, "r") as f:
        for lines in f:
            # strip()方法‌：用于移除字符串头尾的指定字符（默认为空格或换行符）
            # split()方法‌：通过指定的分隔符对字符串进行切片。默认情况下，split()使用任何空白字符作为分隔符，包括空格、制表符、换行符等。
            line = lines.strip().split()
            train_class.append(line[0])
            ys[line[0]] = pos
            pos += 1

    imgs = []

    for x in train_class:
        # opts.data_dir是数据保存的目录，每个类别的数据成一个文件夹，文件夹名是类别名
        cur_class_url=os.path.join(opts.data_dir, x)
        # os.listdir用于列出指定目录下的所有文件和子目录名称。
        # os.path.isfile() 和 os.path.isdir()可以用来判断是文件还是子目录，但这里不需要
        image_list = os.listdir(cur_class_url)  # 这里返回的是该类别目录下的每个样本名称
        for y in image_list:
            path = os.path.join(cur_class_url, y)
            imgs.append((ys[x], path))  # 样本的类别标号，样本的文件路径

    all_dataset={}
    all_dataset["train"]=ImageFolder(imgs, data_transform['train'])
    all_dataset["val"]=ImageFolder(imgs, data_transform['val'])

    L=len(all_dataset["train"])

    all_loader={}
    all_loader["train"] = DataLoader(dataset=all_dataset["train"], batch_size=32, shuffle=True,num_workers=8)
    all_loader["val"] = DataLoader(dataset=all_dataset["val"], batch_size=32, shuffle=False,num_workers=8)

    return all_loader, L


if __name__ == "__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='', help='root of dataset')
    parser.add_argument('--train_class_list', type=str, default='', help='root of Train Class')
    parser.add_argument('--gpu_id', type=int, default=0)
    opts = parser.parse_args()

    device = torch.device(opts.gpu_id)
    print(f'The used device is {device}')
    ResNet = get_ResNet(device)
    # VGG=get_VGG(device)

    criterion = nn.CrossEntropyLoss()  # 微调使用交叉熵损失函数
    optimizer = optim.SGD(ResNet.parameters(), lr=0.001, momentum=0.9)
    # lr_scheduler.StepLR是PyTorch中的一个学习率调度器，用于在训练过程中动态调整学习率。
    # 其主要功能是每隔一定的epoch数（由step_size参数指定），将学习率乘以一个衰减因子（gamma），从而实现对学习率的动态调整。
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    epoch_num = 25

    Loader, length = get_data_loader(opts)

    #############################

    # start=time.time()

    best_model_wts = copy.deepcopy(ResNet.state_dict())  ### save intermediate model
    best_acc = 0.0

    for epoch in range(epoch_num):
        print('Epoch {}/{}'.format(epoch, epoch_num-1))
        print("-"*10)

        for phase in ['train','val']:
            if phase == 'train':
                exp_lr_scheduler.step()
                ResNet.train()
            else:
                ResNet.eval()

            running_loss = 0.0
            running_correct = 0

            for images, labels in Loader[phase]:
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):  # 根据参数的True或False，启用或者禁用梯度

                    outputs = ResNet(images)
                    _, preds = torch.max(outputs, 1)  # 返回张量在指定维度时的最大值及其下标。
                    loss = criterion(outputs, labels)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()*images.size(0)
                running_correct += torch.sum(preds==labels.data)

            epoch_loss = running_loss/length
            epoch_acc = running_correct.double()/length

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase,epoch_loss,epoch_acc))

            if phase == 'val':
                best_model_wts = copy.deepcopy(ResNet.state_dict())
                model_name = "FT_model_"+str(epoch)+".pkl"
                torch.save(best_model_wts, model_name)
