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
from siamese_script import resnet101, normalize_data
import pickle


class EmbedNIFTIDataset(Dataset):
    """
    create a dataset class in PyTorch for reading NIfTI files
    Args:
        source_dir (str): path to images
        transform (Callable): transform to apply to images (Probably None or ToTensor)
        preload (bool): load all data when initializing the dataset
    """

    def __init__(self, data_path, transform=None):
        self.data_fns = np.loadtxt(data_path, dtype=np.str)
        self.transform = transform
        self.data_len = len(self.data_fns)

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        vol_in = self.data_fns[idx]
        main_vol = nib.load(vol_in).get_fdata(dtype=np.float32)
        main_vol = np.moveaxis(main_vol, 1, 0)
        sample = main_vol
        if self.transform is not None:
            sample = self.transform(sample).unsqueeze(0)
        return sample


def call_andrew(data_path='/media/data/Track_2/test_txt.txt', model_path='./model_checkpoints/siamese_final.pth.tar', data_name='all', use_gpu=True):
    assert os.path.exists(data_path)
    assert os.path.exists(model_path)
    dataset = EmbedNIFTIDataset(data_path=data_path,
                                transform=transforms.ToTensor())
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    model = resnet101()
    if use_gpu:
        model = model.cuda()
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load(model_path, map_location='cpu')
        
    model.load_state_dict(checkpoint['model_state_dict'])
    embeds = []
    for i, data in enumerate(data_loader):
        img = data
        if use_gpu:
            img = img.cuda()
        img = normalize_data(img)
        output = model.get_embed(img)
        embeds.append(output.detach().cpu().numpy().reshape(-1))
    np.savetxt(data_name+'embeds.csv', np.array(embeds), delimiter=',', newline='\n')
    return np.array(embeds)

