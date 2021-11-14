import random
import os
import sys
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
#from dataset import *
from test_dataset import *
from torchvision import datasets
from torchvision import transforms
from lenet import *
from discriminator import *
import numpy as np
from function import *
from pretrain_src import *
from train_target import *


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

#target_dataset_name = 'mnistm'
#path = "../../hw3_data/digits"
#checkpoint = "./usps_mnistm_1/"
#target_path = os.path.join(path, target_dataset_name)
cuda = True
batch_size = 128
image_size = 28

if target_dataset_name == 'usps':
    checkpoint = "./svhn_usps"
elif target_dataset_name == 'mnistm':
    checkpoint = "./usps_mnistm_1"
elif target_dataset_name == 'svhn':
    checkpoint = "./mnistm_svhn_1"

if target_dataset_name == 'usps':
    
    transoform = [transforms.ToTensor()]
    
    dataset_target_test = T_Imgdataset(
        target_dir,
        transform=transoform,
        train=False,
        channel=1
    )
else:
    
    transoform = [transforms.ToTensor(), transforms.Normalize(
                                            mean=(0.5,0.5,0.5),
                                            std=(0.5,0.5,0.5))]

    dataset_target_test = T_Imgdataset(
        target_dir,
        transform=transoform,
        train=False
    )

if target_dataset_name == "usps":
    src_encoder = init_model(LeNetEncoder_special())
    src_classifier = init_model(LeNetClassifier_special())
    target_encoder = init_model(LeNetEncoder_special())
    critic = init_model(Discriminator())

else:
    src_encoder = init_model(LeNetEncoder())
    src_classifier = init_model(LeNetClassifier())
    target_encoder = init_model(LeNetEncoder())
    critic = init_model(Discriminator())

if cuda:
    src_encoder = src_encoder.cuda()
    src_classifier = src_classifier.cuda()
    target_encoder = target_encoder.cuda()
    critic = critic.cuda()

print("=== Evaluating classifier for target domain ===")
target_encoder.load_state_dict(torch.load(os.path.join(checkpoint, "target_encoder.ckp")))
src_classifier.load_state_dict(torch.load(os.path.join(checkpoint, "src_classifier.ckp")))
#eval_target(target_encoder, src_classifier, dataset_target_test, cuda)

encoder = target_encoder
classifier = src_classifier

dataloader_target_test = torch.utils.data.DataLoader(
                                dataset=dataset_target_test,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=16
                            )


encoder.eval()
classifier.eval()
n_total = 0
n_correct = 0

output = []

for i, (img, f_name) in enumerate(dataloader_target_test): #(img, label, f_name)
    
    img = Variable(img)
    # label = Variable(label)
    
    if cuda:
        img = img.cuda()
        # label = label.cuda()
    
    preds = classifier(encoder(img))
    
    pred = preds.data.max(1, keepdim=True)[1]
    pred = pred.view((-1, len(img)))
    
    out = pred.squeeze()

    for i in range(len(f_name)):
        output.append([f_name[i], out[i].item()])

    # n_correct += pred.eq(label.data.view_as(pred)).cpu().sum().item()
    # n_total += len(img)

# accu = n_correct * 1.0 / n_total
# print('Accuracy of the %s dataset: %f' % ("target", accu))

import csv
submission = open(output_path, "w+") # "./{}.csv".format(target_dataset_name)
s = csv.writer(submission,delimiter=',',lineterminator='\n')

s.writerow(["image_name","label"])

for i in range(len(output)):
    s.writerow(output[i])
submission.close()