#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     datasets.py
@Time     :     2025/11/20 10:42:58
@Author   :     Louis Swift
@Desc     :     
'''

import numpy as np 
from PIL import Image 
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class MyDataset(Dataset):

    transform = A.Compose([
        A.Resize(512,512),
        A.RandomCrop(256,256),
        ToTensorV2()
    ])  

    def __init__(self,root):
        super().__init__()
        
        self.root = root 
        # 所使用的天气数据集结构为
        # root/rain/*.jpg
        self.paths = list(Path(self.root).glob('**/*.jpg'))

        self.transform = MyDataset.transform
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self,idx):
        path = self.paths[idx]
        image = Image.open(path).convert('RGB')
        image = np.asarray(image)
        image = self.transform(image=image)['image']

        return image 
    
