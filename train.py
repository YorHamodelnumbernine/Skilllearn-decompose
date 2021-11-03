# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 21:48:38 2021

@author: 陈博华
"""
# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
import os.path as osp
from dataselect import makedata
from dataselectstep2 import makedata1

plt.ion()   # interactive mode


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated



def train_model_step1(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



def train_model_step2(modelori,model,net1, num_epochs=25):
    # net1 = nn.Linear(4*3*224*224, 4*1024*28*28)
    net2 = model
    # net3 = nn.Linear(4*1024*28*28,4*3*224*224 )
    # net3 = nn.Linear(1024*28*28,3*224*224 )
    criterion = nn.CrossEntropyLoss()
        
    # # Observe that all parameters are being optimized
    opt1 = optim.SGD(net1.parameters(), lr=args.lr, momentum=0.9)
    opt2 = optim.SGD(net2.parameters(), lr=args.lr, momentum=0.9)
    # opt3 = optim.SGD(net3.parameters(), lr=args.lr, momentum=0.9)
    
    # # Decay LR by a factor of 0.1 every 7 epochs
    scheduler1 = lr_scheduler.StepLR(opt1, step_size=7, gamma=0.1)
    scheduler2 = lr_scheduler.StepLR(opt2, step_size=7, gamma=0.1)
    
    
    since = time.time()

    best_model_wts = copy.deepcopy(net2.state_dict())
    best_acc = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                net2.train()  # Set model to training mode
                net1.train()
            else:
                net2.eval()   # Set model to evaluate 
                net1.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders1[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                opt1.zero_grad()
                # opt3.zero_grad()
                opt2.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    general_representation = get_representation(inputs,net1)
                    # general_representation = nn.functional.interpolate(general_representation, size=[4, 1024, 28, 28],mode='bilinear')
                    # print(general_representation.shape)
                    
                    domain_representation = get_representation(inputs,modelori)
                    # print(domain_representation.shape)
                    inres = general_representation-domain_representation
                    conv = nn.Conv2d(1024, 3, 1).to(device)
                    inres = conv(inres)
                    outputs = net2(inres)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        opt1.step()
                        # opt3.step()
                        opt2.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler1.step()
                scheduler2.step()

            epoch_loss = running_loss / dataset_sizes1[phase]
            epoch_acc = running_corrects.double() / dataset_sizes1[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model        






def visualize_model1(model, num_images=2):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders1['val']):
            inputs = inputs.to(device)
            # print(inputs.shape)
            labels = labels.to(device)
            # print(get_representation(inputs,model).size())
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names1[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def visualize_model(model, num_images=2):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            # print(inputs.shape)
            labels = labels.to(device)
            # print(get_representation(inputs,model).size())
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            print(preds[0])

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                print(class_names)
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)
def get_representation(inputs,model):
    i = 0
    j = 0
    for index,layer in enumerate(model.children()):

        if isinstance(layer, nn.Conv2d):
            img = layer(inputs)
        elif isinstance(layer, nn.Sequential):
            i += 1
            # img = layer(img)
            for index2, layer2 in enumerate(layer):
                if i == 3:
                    j += 1
                img = layer2(img)
                if i == 3 and j == 6:
                    representation = img
                    # print(img.shape)
        elif isinstance(layer, nn.Linear):
            break
    return representation
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='source-target classification')
    parser.add_argument('--method', type=str, default='srconly', choices=['srconly', 'CDAN', 'CDANE', 'DANN',
     'DANNE', 'JAN_Linear', 'JAN', 'DAN_Linear', 'DAN', 'CORAL', 'DDC'])
    parser.add_argument('--net1', type=str, default='resnet50', choices=["resnet18","resnet50", "resnet101"])
    parser.add_argument('--net2', type=str, default='resnet50', choices=["resnet18","resnet50", "resnet101"])
    parser.add_argument('--net3', type=str, default='resnet50', choices=["resnet18","resnet50", "resnet101"])
    parser.add_argument('--dset', type=str, default='office-home', choices=['DomainNet126', 'VISDA-C', 'office', 'office-home'], help="The dataset or source dataset used")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--PATHstep1', type=str, default='/classificationmodel.pt', help="PATHstep1")
    parser.add_argument('--PATHstep2net1', type=str, default='/generalrepmodel.pt', help="PATHstep2net1")
    parser.add_argument('--PATHstep2', type=str, default='/classificationmodel.pt', help="PATHstep2")
    parser.add_argument('--trainstep1', type=bool, default=False, help="train the first step")
    parser.add_argument('--trainstep2', type=bool, default=True, help="train the second step")
    parser.add_argument('--static', type=bool, default=False, help="fix model parameters or not")
    parser.add_argument('--batch_size', type=int, default=2, help="batch_size")
    parser.add_argument('--num_workers',type=int,default=0, help="num_workers")
    parser.add_argument('--max_epoch', type=int, default=25)
    parser.add_argument('--makedata', type=bool, default=False, help="first time make data")
    args = parser.parse_args()
    if args.makedata == True:
        makedata('Clipart','source','train',15)
        makedata('RealWorld','target','train',15)
        makedata('Clipart','source','val',2)
        makedata('RealWorld','target','val',2)
        makedata1('Clipart','train',30)
        makedata1('RealWorld','train',30)
        makedata1('Clipart','val',5)
        makedata1('RealWorld','val',5)
    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65 
    if args.dset == 'office-home':
        data_dir = 'data/source-target'
        data_dir1 = 'data/class/RealWorldclass'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                                 shuffle=True, num_workers=args.num_workers)
                  for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    # print(class_names)
    
    
    image_datasets1 = {y: datasets.ImageFolder(os.path.join(data_dir1, y),
                                              data_transforms[y])
                      for y in ['train', 'val']}
    dataloaders1 = {y: torch.utils.data.DataLoader(image_datasets1[y], batch_size=args.batch_size,
                                                 shuffle=True, num_workers=args.num_workers)
                  for y in ['train', 'val']}
    dataset_sizes1 = {y: len(image_datasets1[y]) for y in ['train', 'val']}
    class_names1 = image_datasets1['train'].classes
    # print(class_names1)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if args.trainstep1:
        if args.net1 == 'resnet18':
            model_domain = models.resnet18(pretrained=True)
        elif args.net1 == 'resnet50':
            model_domain = models.resnet50(pretrained=True)
        elif args.net1 == 'resnet101':
            model_domain = models.resnet101(pretrained=True)
        if args.static:
            for param in model_domain.parameters():
                param.requires_grad = False
        
        # for i,m in enumerate(model_domain.modules()):
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.ReLU) or isinstance(m, nn.MaxPool2d) or isinstance(m, nn.AdaptiveAvgPool2d):
        #         if i == 120:
            
        num_ftrs = model_domain.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model_domain.fc = nn.Linear(num_ftrs, 2)
        
        model_domain = model_domain.to(device)
        
        criterion = nn.CrossEntropyLoss()
        
        # # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_domain.parameters(), lr=args.lr, momentum=0.9)
        
        # # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        
        model_domain = train_model_step1(model_domain, criterion, optimizer_ft, exp_lr_scheduler,
                                num_epochs=args.max_epoch)
        
        torch.save(model_domain, args.PATHstep1) 
        
        visualize_model(model_domain)
    elif args.trainstep2:
        if args.net2 == 'resnet18':
            model_general = models.resnet18(pretrained=True).to(device)
        elif args.net2 == 'resnet50':
            model_general = models.resnet50(pretrained=True).to(device)
        elif args.net2 == 'resnet101':
            model_general = models.resnet101(pretrained=True).to(device)
        num_ftrs = model_general.fc.in_features
        
        if args.net3 == 'resnet18':
            net1 = models.resnet18(pretrained=True).to(device)
        elif args.net3 == 'resnet50':
            net1 = models.resnet50(pretrained=True).to(device)
        elif args.net3 == 'resnet101':
            net1 = models.resnet101(pretrained=True).to(device)
        num_ftrs = model_general.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model_general = model_general.to(device)
        
        model_domain = torch.load(args.PATHstep1)
        for child in model_domain.children():
            for param in child.parameters():
                param.requires_grad=False
                
        try:
            net1 = torch.load(args.PATHstep2net1)
        except:
            pass
        try:
            model_general = torch.load(args.PATHstep2)
        except:
            pass
        model_general = train_model_step2(model_domain,model_general,net1,
                                num_epochs=args.max_epoch)
        
        torch.save(net1, args.PATHstep2net1)
        torch.save(model_general, args.PATHstep2) 
        
        visualize_model1(model_general)
    else :
        model_domain = torch.load(args.PATHstep1)
        # print(model_domain)
        # for i,m in enumerate(model_domain.modules()):
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.ReLU) or isinstance(m, nn.MaxPool2d) or isinstance(m, nn.AdaptiveAvgPool2d):
        #         print(m)
        visualize_model(model_domain)
        model_general = torch.load(args.PATHstep2)
        visualize_model1(model_general)

