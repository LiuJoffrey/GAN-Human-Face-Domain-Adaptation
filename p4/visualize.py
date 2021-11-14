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
from lenet import *
from discriminator import *
from function import *

source_dataset_name = 'mnistm'
target_dataset_name = 'svhn'
print(source_dataset_name, "=>", target_dataset_name)
path = "../../hw3_data/digits"

source_path = os.path.join(path, source_dataset_name)
target_path = os.path.join(path, target_dataset_name)

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

    dataset_source_test = Imgdataset(
        os.path.join(source_path, "test"),
        transform=transoform,
        train=False,
        channel=1
    )
    
    dataset_target_test = Imgdataset(
        os.path.join(target_path, "test"),
        transform=transoform,
        train=False,
        channel=1
    )
else:
    
    transoform = [transforms.ToTensor(), transforms.Normalize(
                                            mean=(0.5,0.5,0.5),
                                            std=(0.5,0.5,0.5))]
    dataset_source_test = Imgdataset(
        os.path.join(source_path, "test"),
        transform=transoform,
        train=False
    )
    dataset_target_test = Imgdataset(
        os.path.join(target_path, "test"),
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

print("=== Load encoder for source and target domain ===")
target_encoder.load_state_dict(torch.load(os.path.join(checkpoint, "target_encoder.ckp")))
src_encoder.load_state_dict(torch.load(os.path.join(checkpoint, "src_encoder.ckp")))

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

if cuda:
    target_encoder = target_encoder.cuda()
    src_encoder = src_encoder.cuda()

### Eval test date ###
target_encoder.eval()
src_encoder.eval()

data_source_iter = iter(dataloader_source)
data_target_iter = iter(dataloader_target)

len_source_dataloader = len(dataloader_source)
len_target_dataloader = len(dataloader_target)

all_latent_space = []
all_class_label = []


for i in range(len_target_dataloader):
    data_target = data_target_iter.next()
    input_img, class_label = data_target
    
    #all_class_label.append(torch.tensor([0]*len(class_label)))
    all_class_label.append(class_label)
    
    if cuda:
        input_img = input_img.cuda()

    inputv_img = Variable(input_img)

    latent_output = target_encoder(inputv_img)
    all_latent_space.append(latent_output.data.cpu())

for i in range(len_source_dataloader):
    data_source = data_source_iter.next()
    input_img, class_label = data_source
    #all_class_label.append(torch.tensor([1]*len(class_label)))
    all_class_label.append(class_label)
    
    if cuda:
        input_img = input_img.cuda()

    inputv_img = Variable(input_img)

    latent_output = src_encoder(inputv_img)
    all_latent_space.append(latent_output.data.cpu())


all_latent_space = torch.cat(all_latent_space).data.numpy()
all_class_label = torch.cat(all_class_label).data.numpy()

print(all_latent_space.shape)
print(all_class_label.shape)

all_cate = {'0':[],'1':[],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[]}
#all_cate = {'0':[],'1':[]}
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

    label = l

    plt.scatter(data[:, 0], data[:, 1], label=label)
plt.legend(prop={'size':5})
plt.show()


exit()
