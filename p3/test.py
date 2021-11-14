import random
import os
import sys
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from test_dataset import *
from torchvision import datasets
from torchvision import transforms
from model import *
import numpy as np

arg = sys.argv

target_dir = arg[1]
target_dataset_name = arg[2]
output_path = arg[3]

if target_dir[0] == '/':
    target_dir = target_dir
else:
    target_dir = os.path.join("../", target_dir)
if output_path[0] == '/':
    output_path = output_path
else:
    output_path = os.path.join("../", output_path)

#target_dataset_name = 'usps'
#path = "../../hw3_data/digits"

#target_path = os.path.join(path, target_dataset_name)


cuda = True
transoform = [transforms.ToTensor()]
dataset_target_test = T_Imgdataset(
    target_dir,
    transform=transoform,
    train=False
)
model = DANN()

if target_dataset_name == 'mnistm':
    model.load_state_dict(torch.load("./usps_mnistm_model_42.pth"))
elif target_dataset_name == 'usps':
    model.load_state_dict(torch.load("./svhn_usps_model.pth"))
elif target_dataset_name == 'svhn':
    model.load_state_dict(torch.load("./mnistm_svhn_model.pth"))


if cuda:
    model = model.cuda()


### Eval test date ###
batch_size = 128
dataloader_target = torch.utils.data.DataLoader(
                            dataset=dataset_target_test,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=16
                        )

len_dataloader = len(dataloader_target)

model.eval()

data_target_iter = iter(dataloader_target)
n_total = 0
n_correct = 0
output = []

for i in range(len_dataloader):
    data_target = data_target_iter.next()
    input_img, f_name = data_target
    
    #class_label = class_label.type(torch.LongTensor)

    if cuda:
        input_img = input_img.cuda()
        # class_label = class_label.cuda()

    inputv_img = Variable(input_img)
    # classv_label = Variable(class_label)

    class_output, _ = model(inputv_img, 0)
    pred = class_output.data.max(1, keepdim=True)[1]
    pred = pred.view((-1, len(input_img)))

    out = pred.squeeze()
    for i in range(len(f_name)):
        output.append([f_name[i], out[i].item()])
    #print(pred)
    #print(classv_label.view_as(pred))
    
    
    # n_correct += pred.eq(classv_label.data.view_as(pred)).cpu().sum().item()
    #print(n_correct)
    
    # n_total += len(input_img)

# accu = n_correct * 1.0 / n_total

# print('accuracy of the %s dataset: %f' % (target_dataset_name, accu))

import csv
submission = open(output_path, "w+") # "./{}.csv".format(target_dataset_name)
s = csv.writer(submission,delimiter=',',lineterminator='\n')

s.writerow(["image_name","label"])

for i in range(len(output)):
    s.writerow(output[i])
submission.close()

