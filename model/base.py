#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     base.py
@Time     :     2025/11/19 19:30:47
@Author   :     Louis Swift
@Desc     :     存放以下四种网络结构（不以类形式构造）

    DSP 的 网络结构 如下：

    1. Encoder 
    2. Decoder 
    3. AdaIN 层 [不在该文件中]
    4. Domain-VAE : fc_encoder & fc_decoder

'''

__all__ = [
    'encoder',
    'decoder',
    'fc_encoder',
    'fc_decoder',
]

import torch.nn as nn 


# Encoder [Actually is VGG]
encoder = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    # Deprecated by Luyanlong, Cuz the relavent weights have been deprecated
    # nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.Conv2d(512, 512, (3, 3)),
    # nn.ReLU(),  # relu4-2
    # nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.Conv2d(512, 512, (3, 3)),
    # nn.ReLU(),  # relu4-3
    # nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.Conv2d(512, 512, (3, 3)),
    # nn.ReLU(),  # relu4-4
    # nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    # nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.Conv2d(512, 512, (3, 3)),
    # nn.ReLU(),  # relu5-1
    # nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.Conv2d(512, 512, (3, 3)),
    # nn.ReLU(),  # relu5-2
    # nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.Conv2d(512, 512, (3, 3)),
    # nn.ReLU(),  # relu5-3
    # nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.Conv2d(512, 512, (3, 3)),
    # nn.ReLU()  # relu5-4
    
)

# Decoder
decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

# Domain-VAE's Encoder 
# ATTEN: :512 - mean ; 512: - std
fc_encoder = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 1024),      
        nn.ReLU(inplace=True),
        nn.Linear(1024, 1024)
)

# Domain-VAE's Decoder
# ATTEN: :512 - mean ; 512: - std
fc_decoder = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 1024)
)
