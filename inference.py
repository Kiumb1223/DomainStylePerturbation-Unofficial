#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     inference.py
@Time     :     2025/11/20 15:42:05
@Author   :     Louis Swift
@Desc     :     I]plement the minimal Inferesnce pipeline
                Input  : single Img with targeted style Img, 
                Output : result Img  
'''

import os
import torch 
import argparse 
import numpy as np 
from PIL import Image 
from loguru import logger 
from model.net import DSPNet
import torch.nn.functional as F
from omegaconf import OmegaConf
import torchvision.utils as vutils
from utils.datasets import transform

def parse_args():
    parser = argparse.ArgumentParser(description='Plz Specify the input image path.')
    parser.add_argument('--gpu_id',type=int,default=0,help='GPU ID.')
    parser.add_argument('-cfg','--config',type=str,default='config/model/DSP.yaml',help='PATH to the Configuration of DSP.')
    parser.add_argument('-c','--content',type=str,required=True,help='PATH to the content image.')
    # parser.add_argument('-s','--style',type=str,required=True,help='PATH to the style image.')
    parser.add_argument('-o','--output',type=str,required=True,help='Directory path to the output image.')

    args = parser.parse_args()

    return args

@torch.no_grad()
def main(args):
    # 1. 获取参数
    device = torch.device(f'cuda:{args.gpu_id}') if torch.cuda.is_available() else torch.device('cpu')
    path_content = args.content
    # path_style = args.style
    path_output = args.output
    os.makedirs(path_output,exist_ok=True)

    # 2. 模型初始化
    cfg_model = OmegaConf.load(args.config)
    model = DSPNet(cfg_model)
    model.to(device)
    model.eval()

    # 3. 模型推理
    content_image = Image.open(path_content).convert('RGB')
    # style_image = Image.open(path_style).convert('RGB')
    
    # 3.1 图像预处理
    content_img_np = np.asarray(content_image)
    ori_H,ori_W,_ = content_img_np.shape
    content_img_th = transform(content_img_np)
    content_img_th = content_img_th.unsqueeze(0).to(device)

    # 3.2 随机采样风格因子
    style_factor = torch.randn(1,512).to(device)
    
    # 3.3 前向传播
    transfer_img = model(content_img_th,style_factor)

    resized_transfer_img = F.interpolate(
        transfer_img,
        size=(ori_H,ori_W),
        mode='bilinear',
        align_corners=False
    )

    # 3.4 保存图像
    name_img = path_content.split(os.sep)[-1].split('.')[0]

    vutils.save_image(
        resized_transfer_img,
        os.path.join(path_output,name_img + '_transfered.png')
    )

if __name__ == '__main__':
    args = parse_args()
    main(args)