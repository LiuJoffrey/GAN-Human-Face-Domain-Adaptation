import numpy as np
import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from dataset import *
from torchvision import datasets
from torchvision import transforms
from visual_model import *

#source_dataset_name = 'usps'
#target_dataset_name = 'mnistm'
source_dataset_name = 'svhn'
target_dataset_name = 'usps'
path = "../../hw3_data/digits"

print(source_dataset_name, "->", target_dataset_name)

source_path = os.path.join(path, source_dataset_name)
target_path = os.path.join(path, target_dataset_name)

cuda = True
cudnn.benchmark = True
batch_size = 128
image_size = 28

transoform = [transforms.ToTensor()]
dataset_source_test = Imgdataset(
    os.path.join(source_path, "test"),
    #os.path.join(source_path, "test.csv"),
    transform=transoform,
    train=False
)
dataset_target_test = Imgdataset(
    os.path.join(target_path, "test"),
    #os.path.join(target_path, "test.csv"),
    transform=transoform,
    train=False
)

model = DANN()
if target_dataset_name == 'mnistm':
    model.load_state_dict(torch.load("./checkpoint/usps_mnistm_model_42.pth"))
elif target_dataset_name == 'usps':
    model.load_state_dict(torch.load("./checkpoint/svhn_usps_model.pth"))
elif target_dataset_name == 'svhn':
    model.load_state_dict(torch.load("./checkpoint/mnistm_svhn_model.pth"))
else:
    print("No such pretrained model task")
    exit()

if cuda:
    model.cuda()

### Eval test date ###
model.eval()
batch_size = 128
dataloader_source = torch.utils.data.DataLoader(
                            dataset=dataset_source_test,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=16
                        )
dataloader_target = torch.utils.data.DataLoader(
                            dataset=dataset_target_test,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=16
                        )
data_source_iter = iter(dataloader_source)
data_target_iter = iter(dataloader_target)

len_source_dataloader = len(dataloader_source)
len_target_dataloader = len(dataloader_target)

all_latent_space = []
all_class_label = []

for i in range(len_target_dataloader):
    data_target = data_target_iter.next()
    input_img, class_label = data_target
    
    all_class_label.append(torch.tensor([0]*len(class_label)))
    #all_class_label.append(class_label)
    
    if cuda:
        input_img = input_img.cuda()

    inputv_img = Variable(input_img)

    latent_output = model(inputv_img, 0)
    all_latent_space.append(latent_output.data.cpu())
#exit()

for i in range(len_source_dataloader):
    data_source = data_source_iter.next()
    input_img, class_label = data_source
    all_class_label.append(torch.tensor([1]*len(class_label)))
    #all_class_label.append(class_label)
    
    if cuda:
        input_img = input_img.cuda()

    inputv_img = Variable(input_img)

    latent_output = model(inputv_img, 0)
    all_latent_space.append(latent_output.data.cpu())



all_latent_space = torch.cat(all_latent_space).data.numpy()
all_class_label = torch.cat(all_class_label).data.numpy()

print(all_latent_space.shape)
print(all_class_label.shape)

#all_cate = {'0':[],'1':[],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[]}

all_cate = {'0':[],'1':[]}

#all_color = ['b','g','r','m','y','k','b','b','b','b',]

"""
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, verbose=1, random_state=300, n_iter=1000, perplexity=20)
dim2 = tsne.fit_transform(all_latent_space)
print(dim2.shape)

outfile = source_dataset_name+"_"+target_dataset_name
np.save(outfile, dim2)
"""
outfile = source_dataset_name+"_"+target_dataset_name
dim2 = np.load(outfile+".npy")
print(dim2.shape)

for i in range(len(dim2)):
    l = str(all_class_label[i])
    all_cate[l].append(dim2[i])

import matplotlib.pyplot as plt
import matplotlib
plt.figure()

for i in range(len(all_cate)):
    l = str(i)
    data = np.array(all_cate[l])
    #print(data[:2,0])
    #exit()
    if i == 0:
        label = "Target_domain"
    else:
        label = "Source_domain"

    #label = l

    plt.scatter(data[:, 0], data[:, 1], label=label)
plt.legend(prop={'size':5})
plt.show()



