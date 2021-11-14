import os
import sys
import cv2
import torch
import numpy as np
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
            
            nn.ConvTranspose2d( 101, figsize * 8, 4, 1, 0, bias=False),
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
            
    def forward(self, X):
        output = self.decoder(X)/2.0+0.5
        return output
    
arg = sys.argv
output_path = arg[1]

if output_path[0] == "/":
    output_path = output_path
else:
    output_path = os.path.join("../", output_path)



### Load Model ###
AC_G = Generator()
AC_G.cuda()
pretrained_acgan = "./G_ACG2_model.pkt.196"
AC_G.load_state_dict(torch.load(pretrained_acgan))
AC_G.eval()

output1 = []
output2 = []
random_seed = [1046, 33,45,50,60,84,169,120,1065,9093]
for seed in random_seed[:]:
    random.seed((seed))
    torch.manual_seed(seed)
    # use for random generation
    up = np.ones(1)
    down = np.zeros(1)
    fixed_class = np.hstack((up,down))
    fixed_class = torch.from_numpy(fixed_class).view(2,1,1,1).type(torch.FloatTensor)
    fixed_noise = torch.randn(1, 100, 1, 1)
    fixed_noise = torch.cat((fixed_noise,fixed_noise))
    fixed_input = Variable(torch.cat((fixed_noise, fixed_class),1)).cuda()

    fixed_img_output = AC_G(fixed_input)
    
    output1.append(fixed_img_output[0].data.unsqueeze(0))
    output2.append(fixed_img_output[1].data.unsqueeze(0))

output_pair_1 = torch.cat(output1)
output_pair_2 = torch.cat(output2)
print(output_pair_1.size())
print(output_pair_2.size())

output = torch.cat((output_pair_1, output_pair_2))
    
torchvision.utils.save_image(output, os.path.join(output_path, "fig2_2.jpg"),nrow=10)