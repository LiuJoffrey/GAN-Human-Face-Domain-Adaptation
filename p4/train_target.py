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


def train_target(src_encoder, src_classifier, target_encoder, critic, 
                        dataset_source_train, dataset_target_train, dataset_target_test, n_epoch=2000, cuda=True, checkpoint="./checkpoint"):
    batch_size = 128
    
    criterion = nn.CrossEntropyLoss()
    if cuda:
        criterion = criterion.cuda()

    optimizer_tgt = optim.Adam(target_encoder.parameters(),
                               lr=1e-4,
                               betas=(0.5, 0.9))
    optimizer_critic = optim.Adam(critic.parameters(),
                               lr=1e-4,
                               betas=(0.5, 0.9))

    dataloader_source_train = torch.utils.data.DataLoader(
                                dataset=dataset_source_train,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=16
                            )
    dataloader_target_train = torch.utils.data.DataLoader(
                                dataset=dataset_target_train,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=16
                            )
    dataloader_target_test = torch.utils.data.DataLoader(
                                dataset=dataset_target_test,
                                batch_size=128,
                                shuffle=True,
                                num_workers=16
                            )

    len_data_loader = min(len(dataloader_source_train), len(dataloader_target_train))

    best_acc = 0
    best_epoch = 0
    
    for epoch in range(n_epoch):
    
        target_encoder.train()
        critic.train()
        n_total = 0
        n_correct = 0
        for i, ((img_src, _), (img_target, _)) in enumerate(zip(dataloader_source_train, dataloader_target_train)):
            
            ### Train domain discriminator ###
            optimizer_critic.zero_grad()

            img_src = Variable(img_src)
            img_target = Variable(img_target)

            if cuda:
                img_src = img_src.cuda()
                img_target = img_target.cuda()
            

            feat_src = src_encoder(img_src)
            feat_target = target_encoder(img_target)
            feat_concat = torch.cat((feat_src, feat_target), 0)

            pred_concat = critic(feat_concat.detach())

            label_src = torch.ones(feat_src.size()[0]).type(torch.LongTensor)
            label_target = torch.zeros(feat_target.size()[0]).type(torch.LongTensor)
            #label_src = torch.ones(feat_src.size()[0]).type(torch.FloatTensor)
            #label_target = torch.zeros(feat_target.size()[0]).type(torch.FloatTensor)
            label_concat = torch.cat((label_src, label_target), 0)

            if cuda:
                label_concat = label_concat.cuda()
            
            loss_critic = criterion(pred_concat, label_concat)
            loss_critic.backward()
            optimizer_critic.step()

            pred_cls = pred_concat.data.max(1, keepdim=True)[1]
            pred_cls = pred_cls.view((-1, len(feat_concat)))
            
            #pred_cls = pred_concat > 0.5
            #pred_cls = pred_cls.type(torch.LongTensor)
            #print(pred_cls)
            #print(label_concat.data.view_as(pred_cls))
            #exit()
            n_correct += pred_cls.eq(label_concat.data.view_as(pred_cls)).cpu().sum().item()
            n_total += len(feat_concat)

            acc = n_correct * 1.0 / n_total

            optimizer_critic.zero_grad()
            ### Train target encoder ### 
            for _ in range(2):
                optimizer_tgt.zero_grad()

                feat_tgt = target_encoder(img_target)
                pred_tgt = critic(feat_tgt)

                label_tgt = torch.ones(pred_tgt.size()[0]).type(torch.LongTensor)
                #label_tgt = torch.ones(pred_tgt.size()[0]).type(torch.FloatTensor)
                if cuda:
                    label_tgt = label_tgt.cuda()

                loss_tgt = criterion(pred_tgt, label_tgt)
                loss_tgt.backward()
                optimizer_tgt.step()

            print("Epoch {}, Step [{}/{}]:"
                      "d_loss={:.5f} g_loss={:.5f} d_acc={:.5f} t_acc={:.5f}"
                      .format(epoch + 1,
                              i + 1,
                              len_data_loader,
                              loss_critic.item(),
                              loss_tgt.item(),
                              acc, best_acc))

        ### Eval the model ###
        

        target_encoder.eval()
        src_classifier.eval()

        n_total = 0
        n_correct = 0

        for i, (img, label) in enumerate(dataloader_target_test):

            img = Variable(img)
            label = Variable(label)
            if cuda:
                img = img.cuda()
                label = label.cuda()
            
            preds = src_classifier(target_encoder(img))

            pred = preds.data.max(1, keepdim=True)[1]
            pred = pred.view((-1, len(img)))
            n_correct += pred.eq(label.data.view_as(pred)).cpu().sum().item()

            n_total += len(img)
        
        accu = n_correct * 1.0 / n_total
        print('epoch: %d, accuracy of the %s dataset: %f' % (epoch, "target", accu))

        if accu > best_acc:
            print("New record Achieve")
            best_acc = accu
            best_epoch = epoch
            torch.save(target_encoder.state_dict(), os.path.join(checkpoint, "target_encoder.ckp"))
            #torch.save(src_classifier.state_dict(), os.path.join("checkpoint", "final_classifier.ckp"))
    
    print(best_epoch, ": ", best_acc)
    return target_encoder


def eval_target(encoder, classifier, dataset_target_test, cuda=True):
    batch_size = 128
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
    for i, (img, label) in enumerate(dataloader_target_test):
        
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
    print('Accuracy of the %s dataset: %f' % ("target", accu))



