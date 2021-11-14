import os
import sys
import cv2
import  torch
import  numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import random
import pickle
from tqdm import tqdm
import csv

class Generator(nn.Module):
    def __init__(self, figsize=64):
        super(Generator, self).__init__()
        self.decoder = nn.Sequential(
            
            nn.ConvTranspose2d(100, figsize * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(figsize * 8),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(figsize * 8, figsize * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(figsize * 4),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(figsize * 4, figsize * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(figsize * 2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(figsize * 2, figsize, 4, 2, 1, bias=False),
            nn.BatchNorm2d(figsize),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(figsize, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            
        )

    def forward(self, data):
        output = self.decoder(data)/2.0+0.5
        return output


arg = sys.argv
output_path = arg[1]
if output_path[0] == "/":
    output_path = output_path
else:
    output_path = os.path.join("../", output_path)

###
G = Generator()
G.cuda()
pretrained_acgan = "./G_model.pkt.105"
G.load_state_dict(torch.load(pretrained_acgan))
G.eval()

random_seed = [72, 76, 88, 110, 159, 191, 198, 213,
               229, 249, 303, 316, 338, 339, 348, 355, 
               363, 393, 414, 427, 430, 441, 447, 448, 
               547, 553, 555, 583, 632, 633, 673, 702]
output = []
for seed in random_seed:
    
    random.seed((seed))
    torch.manual_seed(seed)
    fixed_noise = torch.randn(1, 100, 1, 1)
    fixed_noise = Variable(fixed_noise).cuda()
    
    fixed_img_output = G(fixed_noise)
    output.append(fixed_img_output.cpu().data)
output = torch.cat(output, 0)
torchvision.utils.save_image(output, os.path.join(output_path, "fig1_2.jpg"),nrow=8)