#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from torch.utils.data import Dataset, DataLoader
from torchvision import utils, transforms
from glob import glob
import random
import os
from torch import optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
# import horovod.torch as hvd


# In[2]:


class MasterNIFTIDataset(Dataset):
    """
    create a dataset class in PyTorch for reading NIfTI files
    Args:
        source_dir (str): path to images
        transform (Callable): transform to apply to images (Probably None or ToTensor)
        preload (bool): load all data when initializing the dataset
    """

    def __init__(self, positive_dir, negative_dir, transform=None):
        self.pos_fns = sorted(glob(os.path.join(positive_dir, "*.nii.gz")))
        self.neg_fns = sorted(glob(os.path.join(negative_dir, "*.nii.gz")))
        self.transform = transform
        self.pos_len = len(self.pos_fns)

    def __len__(self):
        return self.pos_len+len(self.neg_fns)

    def __getitem__(self, idx):
        neg_img = False
        if idx >= self.pos_len:
            neg_img = True
            idx -= self.pos_len
            vol_in = self.neg_fns[idx]
        else:
            vol_in = self.pos_fns[idx]

        same_class = random.randint(0,1)
        if (same_class and neg_img) or (not same_class and not neg_img):
            fn = random.choice(self.neg_fns)
        elif (same_class and not neg_img) or (not same_class and neg_img):
            fn = random.choice(self.pos_fns)
        main_vol = nib.load(vol_in).get_fdata(dtype=np.float32)
        partner_vol = nib.load(fn).get_fdata(dtype=np.float32)
        main_vol = np.moveaxis(main_vol, 1, 0)
        partner_vol = np.moveaxis(partner_vol, 1, 0)
        sample = [main_vol, partner_vol]
        if self.transform is not None:
            sample = [self.transform(sample[0]).unsqueeze(0), self.transform(sample[1]).unsqueeze(0)]
        if same_class:
            sample.append(torch.Tensor([0]))
        else:
            sample.append(torch.Tensor([1]))
        return sample


# In[3]:


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive


# In[4]:


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 cardinality=32,
                 num_classes=400):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(1, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
                                       cardinality)
        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.layer4 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        last_duration = int(math.ceil(sample_duration / 19))  # hacked?
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size+1, last_size), stride=1)  # hacked also
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)
    
    def get_embed(self, data_in):
        data_in = self.conv1(data_in)
        data_in = self.bn1(data_in)
        data_in = self.relu(data_in)
        data_in = self.maxpool(data_in)
        data_in = self.layer1(data_in)
        data_in = self.layer2(data_in)
        data_in = self.layer3(data_in)
        data_in = self.layer4(data_in)
        data_in = self.avgpool(data_in)
        data_in = data_in.view(data_in.size(0), -1)

        return data_in

    def model_pass(self, data_in):
        data_in = self.conv1(data_in)
        data_in = self.bn1(data_in)
        data_in = self.relu(data_in)
        data_in = self.maxpool(data_in)
        data_in = self.layer1(data_in)
        data_in = self.layer2(data_in)
        data_in = self.layer3(data_in)
        data_in = self.layer4(data_in)
        data_in = self.avgpool(data_in)
        data_in = data_in.view(data_in.size(0), -1)
        data_in = self.fc(data_in)

        return data_in
    
    def forward(self, x_arr_in):
        labels = []
        for sample in x_arr_in:
            labels.append(self.get_embed(sample))
        return labels


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], 
                    num_classes=2,
                    shortcut_type='B',
                    cardinality=32,
                    sample_size=182,
                    sample_duration=218)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], 
                    num_classes=2,
                    shortcut_type='B',
                    cardinality=32,
                    sample_size=182,
                    sample_duration=218)
    return model


# In[5]:


# Important functions
def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.savefig('loss_fig.png')
    plt.show()

def normalize_data(images_batch):
    mean_val = images_batch.mean(dim=[2,3,4])
    std_val = images_batch.std(dim=[2,3,4])
    images_batch = (images_batch-mean_val[:, :, None, None, None])/std_val[:, :, None, None, None]
    return images_batch


# In[6]:


# Create train and test sets
# pos_fns = sorted(glob(os.path.join('/media/data/Track_2/good', "*.nii.gz")))
# neg_fns = sorted(glob(os.path.join('/media/data/Track_2/bad', "*.nii.gz")))
# validation_split = .1
# shuffle_dataset = True
# random_seed= 42

# Creating data indices for training and validation splits:
# pos_size = len(pos_fns) + len(neg_fns)
# indices = list(range(pos_size))
# split = int(np.floor(validation_split * pos_size))
# if shuffle_dataset :
#     np.random.seed(random_seed)
#     np.random.shuffle(indices)
# train_indices, val_indices = indices[split:], indices[:split]
# train_sampler = SubsetRandomSampler(train_indices)
# val_sampler = SubsetRandomSampler(val_indices)


# In[7]:
if __name__ == "__main__":
    # hvd.init()
    net = resnet101().cuda()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(),lr = 0.1)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)
    counter = []
    loss_history = []
    train_epochs = 5
    iteration_number= 0
    dataset = MasterNIFTIDataset(positive_dir='/media/data/Track_2/good',
                                   negative_dir='/media/data/Track_2/bad',
                                   transform=transforms.ToTensor())

    # train_sampler = torch.utils.data.distributed.DistributedSampler(
    #     dataset, num_replicas=hvd.size(), rank=hvd.rank())

    batch_size = 1
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True)

    # compression = hvd.Compression.fp16
    # optimizer = hvd.DistributedOptimizer(
    #     optimizer, named_parameters=net.named_parameters(),
    #     compression=compression,
    #     backward_passes_per_step=1)

    checkpoint = torch.load(f'/nethome/asilva9/brains/model_checkpoints/siamese_epoch2.pth.tar')
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # hvd.broadcast_parameters(net.state_dict(), root_rank=0)
    # hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    same_class_loss = 0
    same_class_count = 0
    diff_class_loss = 0
    diff_class_count = 0
    for epoch in range(train_epochs):
        for i, data in enumerate(train_loader,0):
            img0, img1 , label = data
            img0 = normalize_data(img0)
            img1 = normalize_data(img1)
            img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
            optimizer.zero_grad()
            output1,output2 = net([img0,img1])
            loss_contrastive = criterion(output1,output2,label)
            if np.isnan(loss_contrastive.item()):
                continue
            loss_contrastive.backward()
            optimizer.step()
    #         if label > 0:
    #             diff_class_loss += loss_contrastive.item()
    #             diff_class_count += 1
    #         else:
    #             same_class_loss += loss_contrastive.item()
    #             same_class_count += 1
            if i % 10 == 0:
                print(i)
            if i %100 == 0 :
                print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
                iteration_number +=100
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
    #             print(f"Average diff class loss is {diff_class_loss/max(diff_class_count, 1)}")
    #             print(f"Average same class loss is {same_class_loss/max(same_class_count, 1)}")
                torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_history,
                }, f'/nethome/asilva9/brains/model_checkpoints/siamese_epoch_{epoch}_iter_{iteration_number}.pth.tar')

        torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_history,
                }, f'/nethome/asilva9/brains/model_checkpoints/siamese_epoch_hvd{epoch}.pth.tar')
    #     sim_loss = 0
    #     diff_loss = 0
    #     for i, data in enumerate(validation_loader, 0):
    #         img0, img1 , label = data
    #         img0 = normalize_data(img0)
    #         img1 = normalize_data(img1)
    #         img0, img1 , label = img0.cuda(), img1.cuda() , label.cuda()
    #         output1,output2 = net([img0,img1])
    #         loss_contrastive = criterion(output1,output2,label)
    #         if label > 0:
    #             diff_loss += loss_contrastive.item()
    #         else:
    #             sim_loss += loss_contrastive.item()
    #     print(f"Diff loss: {diff_loss}")
    #     print(f"Sim loss: {sim_loss}")

    show_plot(counter,loss_history)


    # In[ ]:




