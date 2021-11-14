import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from dataset import *
from torchvision import datasets
from torchvision import transforms
from model import *
import numpy as np


source_dataset_name = 'usps'
target_dataset_name = 'mnistm'
path = "../../hw3_data/digits"
print(source_dataset_name, "->", target_dataset_name)

source_path = os.path.join(path, source_dataset_name)
target_path = os.path.join(path, target_dataset_name)

cuda = True
cudnn.benchmark = True
lr = 1e-3
batch_size = 128
image_size = 28
n_epoch = 100

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

transoform = [transforms.ToTensor()]

### source data ###
dataset_source_train = Imgdataset(
    os.path.join(source_path, "train"),
    #os.path.join(source_path, "train.csv"),
    transform=transoform,
    train=True
)

dataset_source_test = Imgdataset(
    os.path.join(source_path, "test"),
    #os.path.join(source_path, "test.csv"),
    transform=transoform,
    train=False
)


### target data ###
dataset_target_train = Imgdataset(
    os.path.join(target_path, "train"),
    #os.path.join(target_path, "train.csv"),
    transform=transoform,
    train=True
)

dataset_target_test = Imgdataset(
    os.path.join(target_path, "test"),
    #os.path.join(target_path, "test.csv"),
    transform=transoform,
    train=False
)


model = DANN()
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_class = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()

if cuda:
    model.cuda()
    loss_class.cuda()
    loss_domain.cuda()

log = {}
log["target_acc"] = []

best_acc = 0
best_epoch = 0
for epoch in range(n_epoch):
    batch_size = 128
    ### Training ###
    dataloader_source = torch.utils.data.DataLoader(
                                dataset=dataset_source_train,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=16
                            )
    dataloader_target = torch.utils.data.DataLoader(
                                dataset=dataset_target_train,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=16
                            )
    
    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    

    model.train()

    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)


    train_loss = 0
    for i in range(len_dataloader):
        p = float(i + epoch * len_dataloader) / (n_epoch *len_dataloader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        model.zero_grad()

        ### train source domain ###
        data_source = data_source_iter.next()
        input_img, class_label = data_source

        batch_size = len(class_label)
        class_label = class_label.type(torch.LongTensor)
        domain_label = torch.zeros(batch_size)
        domain_label = domain_label.type(torch.LongTensor)

        if cuda:
            input_img = input_img.cuda()
            class_label = class_label.cuda()
            domain_label = domain_label.cuda()

        inputv_img = Variable(input_img)
        classv_label = Variable(class_label)
        domainv_label = Variable(domain_label)

        class_output, domain_output = model(inputv_img, alpha)
        
        err_s_label = loss_class(class_output, classv_label)
        err_s_domain = loss_domain(domain_output, domainv_label)
        
        ### train target domain ###
        data_target = data_target_iter.next()
        input_img, _ = data_target

        batch_size = len(input_img)
        domain_label = torch.ones(batch_size)
        domain_label = domain_label.type(torch.LongTensor)

        if cuda:
            input_img = input_img.cuda()
            domain_label = domain_label.cuda()

        inputv_img = Variable(input_img)
        domainv_label = Variable(domain_label)

        _, domain_output = model(inputv_img, alpha)
        err_t_domain = loss_domain(domain_output, domainv_label)

        err = err_t_domain + err_s_domain + err_s_label
        err.backward()
        #err_s_label.backward()
        optimizer.step()

        train_loss += err_s_label.item()
        
        print('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
              % (epoch, i, len_dataloader, err_s_label.item(),
                 err_s_domain.item(), err_t_domain.item()))
        
    
    ### Eval test date ###
    dataloader_target = torch.utils.data.DataLoader(
                                dataset=dataset_target_test,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=16
                            )
    
    dataloader_source = torch.utils.data.DataLoader(
                                dataset=dataset_source_test,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=16
                            )

    len_dataloader = len(dataloader_target)
    #len_dataloader = len(dataloader_source)
    
    model.eval()

    data_target_iter = iter(dataloader_target)
    #data_target_iter = iter(dataloader_source)
    n_total = 0
    n_correct = 0
    for i in range(len_dataloader):
        data_target = data_target_iter.next()
        input_img, class_label = data_target
        
        #class_label = class_label.type(torch.LongTensor)

        if cuda:
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        inputv_img = Variable(input_img)
        classv_label = Variable(class_label)

        class_output, _ = model(inputv_img, 0)
        pred = class_output.data.max(1, keepdim=True)[1]
        pred = pred.view((-1, len(input_img)))
        #print(pred)
        #print(classv_label.view_as(pred))
        
        
        n_correct += pred.eq(classv_label.data.view_as(pred)).cpu().sum().item()
        #print(n_correct)
        
        n_total += len(input_img)
    
    accu = n_correct * 1.0 / n_total
    log["target_acc"].append(accu)
    print('epoch: %d, accuracy of the %s dataset: %f' % (epoch, "target", accu))

    if accu > best_acc:
        print("New record Achieve")
        best_acc = accu
        best_epoch = epoch
        torch.save(model.state_dict(), '{0}/Train_{1}_{2}_model.pth'.format("checkpoint", source_dataset_name, target_dataset_name))

print(best_epoch, ": ", best_acc)