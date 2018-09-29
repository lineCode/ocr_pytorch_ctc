#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import freetype
import copy
import random
import cv2
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import torchvision
import torch
from torchvision import datasets
from torchvision import models
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss
from models import crnn_model, cnn_model, lstm_model


class resizeNormalize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


def decode(output):
    pred = []
    preds = []
    for i, p in enumerate(output):
        for t in range(len(p)):
            if p[t] != 0 and ((t == 0) or (t != 0 and p[t] != p[t - 1])):
                pred.append(int(p[t]))
        preds.append(pred)
        pred = []
    return preds


class ImageData(Dataset):
    def __init__(self, root, label_txt, transforms=None):
        with open(os.path.join(root, label_txt), "r") as file:
            lines = file.readlines()
            self.image_names = [os.path.join(root, line.strip().split(" ")[0] + ".png") for line in lines]
            self.image_labels = [line.strip().split(" ")[1] for line in lines]
        self.transforms = transforms

    def __getitem__(self, item):
        image_name = self.image_names[item]
        image_label = self.image_labels[item]
        image = Image.open(image_name).convert("L")
       # image = image.transpose(Image.ROTATE_90)
       # image.show()
        if self.transforms != None:
            image = self.transforms(image)
        label = image_label
        return image, label

    def __len__(self):
        return len(self.image_names)


def label_one_hot(label):
    label_list = list(label)
    label_list = [[int(l)] for l in label_list]
    matric = torch.FloatTensor(18, 10).zero_()
    one_hot = matric.scatter_(dim=1, index=torch.LongTensor(label_list), value=1)
    return one_hot


def label_to_tensor(labels):
    label = labels
    label_list = []
    for label in labels:
        label_list.extend(map(int, list(label)))
    label_list = np.array(label_list) + 1
    label_list = torch.IntTensor(label_list)
    return label_list


def test(model, transform, epoch):
    imageset = ImageData(root="/home/simple/mydemo/ocr_tensorflow_cnn-master/images_labels/val/",
                         label_txt="labels.txt", transforms=transform)
    dataload = DataLoader(dataset=imageset, batch_size=batch_size)
    all_num = 4000
    acc_num = 0.0
    for i, data in enumerate(dataload):
        images = data[0]
        labels = data[1]
        labels = label_to_tensor(labels)
        labels = labels.view(32, 18)
        images = Variable(images).cuda()
        outputs = model(images)
        output = outputs.max(2)[1]
        output = output.transpose(1, 0)
        output = decode(output)

        for index in range(batch_size):
            pred_label = torch.IntTensor(output[index])
            label = labels[index]
            #   print("label:{}......pred_label:{},len={}".format(np.array(label),np.array(pred_label),len(pred_label)))
            if len(pred_label) == len(label):
                num = label.eq(pred_label).sum()
                if num == 18:
                    acc_num += 1
    acc = acc_num / all_num
    print("epoch{}:accuracy={}".format(epoch, acc))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoches = 100
batch_size = 32


def main():
    # transform=resizeNormalize((280,32))
    transform = transforms.ToTensor()
    imageset = ImageData(root="/home/simple/mydemo/ocr_tensorflow_cnn-master/images_labels/train/",
                         label_txt="labels.txt", transforms=transform)
    dataloader = DataLoader(dataset=imageset, batch_size=batch_size, shuffle=True)

    # model_path="./cnn_model.pth"
    # if os.path.exists(model_path):
    #     model.load_state_dict(torch.load(model_path))
    model=lstm_model.LSTM_MODEL().cuda()

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    #optimizer=optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
    #optimizer=optim.RMSprop(model.parameters(),lr=0.01)
    #  optimizer=optim.Adagrad(model.parameters())

    schedular = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5)
    criterion = CTCLoss(blank=0)
    criterion = criterion.cuda()

    for epoch in range(epoches):
        schedular.step()
        for index, data in enumerate(dataloader):
            images = data[0]
            labels = data[1]
            labels = label_to_tensor(labels)
            images = Variable(images).cuda()
            labels = Variable(labels)
            outputs = model(images)
            label_sizes = torch.IntTensor([18] * int(32))
            probs_sizes = torch.IntTensor([64] * int(32))

            loss = criterion(outputs, labels, probs_sizes, label_sizes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if index % 50 == 0:
                print("epoch[{}/{}]:loss={}".format(index, epoch, loss / 32))
            if (index + 1) % 200 == 0:
                 test(model, transform, epoch)


main()

