#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     train.py
@Time     :     2025/11/19 19:46:59
@Author   :     Louis Swift
@Desc     :     
'''

import os 
import hydra
import torch
from loguru import logger 
from model.net import DSPNet
from torch.optim import Adam
from datetime import datetime
from omegaconf import OmegaConf
import torchvision.utils as vutils
from utils.datasets import MyDataset
from utils.logger import setup_logger
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from utils.sampler import InfiniteSamplerWrapper
from torch.utils.tensorboard import SummaryWriter
from utils.misc import set_random_seed,get_model_info


@logger.catch
@hydra.main(config_path='config', config_name='config.yaml',version_base=None)
def main(config:OmegaConf):
    # ----------------------
    # 1. 基础配置
    # 以当前时间步为实验文件夹
    exp_dir = config.exp.exp_dir + os.sep + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(exp_dir,exist_ok=True)

    # 权重保存目录
    ckpt_dir = exp_dir + os.sep + 'checkpoints'
    os.makedirs(ckpt_dir,exist_ok=True)

    # 日志文件目录
    setup_logger(exp_dir)
    # tensorboard 目录
    tb_dir = exp_dir + os.sep + 'tb_logs'
    os.makedirs(tb_dir,exist_ok=True)
    writer = SummaryWriter(tb_dir)
    
    logger.info(f'Exp Directory : {exp_dir}.')
    logger.info(f'Ckpt Directory : {ckpt_dir}.')
    logger.info(f'Tensorboard Directory : {tb_dir}.')

    # 固定随机种子
    set_random_seed(config.exp.seed)

    # ----------------------
    # 2. 模型配置
    net = DSPNet(config.model)
    net.to(config.exp.device)
    net.train()
    logger.info("DSP Model Size:\n" + get_model_info(net))

    # ----------------------
    # 3. 数据集配置
    content_dataset = MyDataset(config.data.content_dir)
    style_dataset = MyDataset(config.data.style_dir)

    content_iter = iter(DataLoader(
        dataset=content_dataset,
        batch_size=config.exp.batch_size,
        pin_memory=True,
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=config.exp.num_workers
    ))

    style_iter = iter(DataLoader(
        dataset=style_dataset,
        batch_size=config.exp.batch_size,
        pin_memory=True,
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=config.exp.num_workers
    ))

    # ----------------------
    # 4. 优化器与调度器
    optimizer = Adam(
        net.parameters(),
        lr=config.exp.lr,weight_decay=config.exp.lr_decay
    )
    def lr_lambda(it):
        """
        it：从 0 开始的 step 计数
        warmup 阶段：线性
        之后： 1 / (1 + lr_decay * it) 衰减
        """
        warmup_iter = getattr(config.exp, "warmup_iter", None)

        if warmup_iter is not None and it < warmup_iter:
            # 线性 warmup
            return float(it + 1) / float(warmup_iter)
        else:
            # lr 衰减策略
            return 1.0 / (1.0 + config.exp.lr_decay * it)
        
    scheduler = LambdaLR(optimizer,lr_lambda=lr_lambda)


    # ----------------------
    # 5. 训练


    logger.info('------------ START TO TRAIN ------------')
    bt_warmup = False

    if config.exp.warmup_iter is not None:
        logger.info(f'Use encoder warmup for {config.exp.warmup_iter} iters.')
        for p in net.decoder.parameters():  
            p.requires_grad = False
        bt_warmup = True 

    for cur_iter in range(config.exp.max_iter):

        if bt_warmup is True and (config.exp.warmup_iter is not None and cur_iter == config.exp.warmup_iter):
            logger.info('Warmup finished. Unfreeze decoder.')
            for p in net.decoder.parameters():
                p.requires_grad = True
            
            bt_warmup = False 

        content_imgs = next(content_iter).to(config.exp.device)
        style_imgs = next(style_iter).to(config.exp.device)

        loss_dict,g_t = net(content_imgs,style_imgs)

        # 损失计算
        loss = config.loss.content_loss * loss_dict['loss_c'] + \
               config.loss.style_loss * loss_dict['loss_s'] + \
               config.loss.kl_loss * loss_dict['loss_KL'] + \
               config.loss.rec_loss * loss_dict['loss_Rec']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        logger.info(f'Iter:[{cur_iter}/{config.exp.max_iter}], Loss:[{loss.item():.3f}].')

        # 保存模型权重
        if (cur_iter + 1) % config.exp.save_model_interval == 0 or (cur_iter + 1) == config.exp.max_iter:
            state_dict = net.state_dict()
            torch.save(state_dict, ckpt_dir + os.sep + f'model_iter_{cur_iter + 1:d}.pth')

            logger.info(f"Model checkpoints saved at {ckpt_dir + os.sep + f'model_iter_{cur_iter + 1:d}.pth'}.")

        # 记录实验数据
        writer.add_scalar('content_loss',loss_dict['loss_c'].item(),cur_iter+1)
        writer.add_scalar('style_loss',loss_dict['loss_s'].item(),cur_iter+1)
        writer.add_scalar('kl_loss',loss_dict['loss_KL'].item(),cur_iter+1)
        writer.add_scalar('rec_loss',loss_dict['loss_Rec'].item(),cur_iter+1)
        writer.add_scalar('loss_total',loss.item(),cur_iter+1)

        if (cur_iter + 1) % config.exp.vis_interval == 0:
            content_show = content_imgs[0].detach().cpu()
            style_show   = style_imgs[0].detach().cpu()
            gt_show      = g_t[0].detach().cpu()

            writer.add_image('grid/content',
                vutils.make_grid(content_imgs, normalize=True, scale_each=True),
                cur_iter+1)

            writer.add_image('grid/style',
                vutils.make_grid(style_imgs, normalize=True, scale_each=True),
                cur_iter+1)

            writer.add_image('grid/g_t',
                vutils.make_grid(g_t, normalize=True, scale_each=True),
                cur_iter+1)

    writer.close()
if __name__ == '__main__':
    main()