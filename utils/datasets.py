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

__all__ = [
    'transform',
    'MyDataset'
]

transform = A.Compose([
    A.Resize(512,512),
    A.RandomCrop(256,256),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])  

class MyDataset(Dataset):


    def __init__(self,root):
        super().__init__()
        
        self.root = root 
        # 所使用的天气数据集结构为
        # root/rain/*.jpg
        self.paths = list(Path(self.root).glob('**/*.jpg'))

        self.transform = transform
    
    def __len__(self):
        return len(self.paths)

    def __getitem__(self,idx):
        path = self.paths[idx]
        image = Image.open(path).convert('RGB')
        image = np.asarray(image)
        image = self.transform(image=image)['image']

        return image 
    
