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
import numpy as np

def train_src(encoder, classifier, dataset_source_train, dataset_source_test, n_epoch_pre=100, cuda=True, pre_src="./checkpoint/"):
    
    batch_size = 128
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=1e-3,
        betas=(0.5, 0.999))

    criterion = nn.CrossEntropyLoss()
    if cuda:
        criterion = criterion.cuda()

    best_acc = 0
    best_epoch = 0

    dataloader_source_train = torch.utils.data.DataLoader(
                                dataset=dataset_source_train,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=16
                            )
    dataloader_source_test = torch.utils.data.DataLoader(
                                dataset=dataset_source_test,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=16
                            )

    best_acc = 0
    best_epoch = 0
    for epoch in range(n_epoch_pre):
        ### Train the model ###
        encoder.train()
        classifier.train()
        len_dataloader = len(dataloader_source_train)

        for i, (img, label) in enumerate(dataloader_source_train):
            optimizer.zero_grad()

            img = Variable(img)
            label = Variable(label)

            if cuda:
                img = img.cuda()
                label = label.cuda()
            
            preds = classifier(encoder(img))
            loss = criterion(preds, label)
            loss.backward()
            optimizer.step()

            print('epoch: %d, [iter: %d / all %d], err_s_label: %f' \
              % (epoch, i, len_dataloader, loss.item()))

        ### Eval the model ###
        encoder.eval()
        classifier.eval()
        n_total = 0
        n_correct = 0
        for i, (img, label) in enumerate(dataloader_source_test):
            
            img = Variable(img)
            label = Variable(label)

            if cuda:
                img = img.cuda()
                label = label.cuda()
            
            preds = classifier(encoder(img))
            
            pred = preds.data.max(1, keepdim=True)[1]
            pred = pred.view((-1, len(img)))
            n_correct += pred.eq(label.data.view_as(pred)).cpu().sum().item()

            n_total += len(img)
        
        accu = n_correct * 1.0 / n_total
        print('epoch: %d, accuracy of the %s dataset: %f' % (epoch, "source", accu))

        if accu > best_acc:
            print("New record Achieve")
            best_acc = accu
            best_epoch = epoch
            torch.save(encoder.state_dict(), os.path.join(pre_src, "src_encoder.ckp"))
            torch.save(classifier.state_dict(), os.path.join(pre_src, "src_classifier.ckp"))
    
    print(best_epoch, ": ", best_acc)
    return encoder, classifier


def eval_src(encoder, classifier, dataset_source_test, cuda=True):
    batch_size = 128
    dataloader_source_test = torch.utils.data.DataLoader(
                                dataset=dataset_source_test,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=16
                            )

    encoder.eval()
    classifier.eval()
    n_total = 0
    n_correct = 0
    for i, (img, label) in enumerate(dataloader_source_test):
        
        img = Variable(img)
        label = Variable(label)

        if cuda:
            img = img.cuda()
            label = label.cuda()
        
        preds = classifier(encoder(img))
        
        pred = preds.data.max(1, keepdim=True)[1]
        pred = pred.view((-1, len(img)))
        n_correct += pred.eq(label.data.view_as(pred)).cpu().sum().item()

        n_total += len(img)
    
    accu = n_correct * 1.0 / n_total
    print('Accuracy of the %s dataset: %f' % ("source", accu))




