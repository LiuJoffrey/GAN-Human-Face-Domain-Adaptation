import cv2
import os
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from scipy.misc import imread, imresize
import random
import csv
import json


train_file_img = "../../hw3_data/digits/svhn/train"


def main():
    
    img_size = 28
    train_data = Imgdataset(train_file_img, img_size, transform=[transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataload = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False)

    test = {}
    for i, batch in enumerate(dataload):
        #print(i)
        if i > 20:
            break
        print(batch[0].size())
        print(type(batch[0]))

class T_Imgdataset(data.Dataset):
    def __init__(self, img_folder, img_size=28, transform=[], train=False):
        self.img_folder = img_folder
        #self.ann_data = img_folder+".csv"
        self.transform = transform
        self.img_size = img_size

        # self.all_ann_label = {}
        # with open(self.ann_data, 'r') as f:
        #     rows = csv.DictReader(f)
        #     for row in rows:
        #         self.all_ann_label[row['image_name']] = int(float(row['label']))
        
        self.all_img_path = []
        all_img_file = os.listdir(img_folder)
        for file_name in all_img_file:
            self.all_img_path.append(file_name)

        
    def __len__(self):
        
        return len(self.all_img_path)   

    def __getitem__(self, index):
        
        img_file_name = self.all_img_path[index]
        # label = self.all_ann_label[img_file_name]
        
        img_path = os.path.join(self.img_folder, img_file_name)
        img = cv2.imread(img_path)
        

        if img.shape[0] != self.img_size:
            print("resize")
            img = cv2.resize(img, (self.img_size, self.img_size))

        img = self.BGR2RGB(img)
        for t in self.transform:
            img = t(img)

        return img, img_file_name


    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


if __name__ == '__main__':
    main()