#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     net.py
@Time     :     2025/11/19 19:43:13
@Author   :     Louis Swift
@Desc     :     

'''


import os 
import copy
import torch
import torch.nn as nn 
from loguru import logger 
import torch.nn.functional as F 

from .base import encoder,decoder,fc_encoder,fc_decoder
from .function import \
        calc_mean_std,\
        calc_feat_mean_std,\
        adaptive_instance_normalization_Domain_VAE as adain

class DSPNet(nn.Module):
    def __init__(self,ckpt_dict):
        super().__init__()

        # 为了计算损失，需要进一步把 encoder 拆开
        enc_layers = nn.ModuleList(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])
        self.enc_2 = nn.Sequential(*enc_layers[4:11])
        self.enc_3 = nn.Sequential(*enc_layers[11:18])
        self.enc_4 = nn.Sequential(*enc_layers[18:31])        

        self.decoder = decoder
        self.fc_encoder = fc_encoder
        self.fc_decoder = fc_decoder

        # Encoder层不参与训练
        # Fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.load_ckpt(ckpt_dict)

        logger.info('Complete the Initialization of [DSPNet].')

    def load_ckpt(self,ckpt_dict):

        ckpt_path = ckpt_dict.get('ckpt_path',None)
        if ckpt_path is not None and os.path.isfile(ckpt_path):
            state = torch.load(ckpt_path,map_location='cpu')
            self.load_state_dict(state)
            logger.info(f'Load ckpt from : {ckpt_path}.')
            return 

        # 载入 encoder 权重

        encoder_path = ckpt_dict.get('encoder_path',None)

        assert encoder_path is not None and os.path.isfile(encoder_path), f'Invaild path :{encoder_path}.'

        encoder_state_dict  = torch.load(encoder_path,map_location='cpu')

        tmp_encoder = copy.deepcopy(encoder)
        tmp_encoder.load_state_dict(encoder_state_dict)

        # 再按同样的切片方式拆成 4 段，并把权重拷到 self.enc_*
        tmp_layers = nn.ModuleList(tmp_encoder.children())

        tmp_enc_1 = nn.Sequential(*tmp_layers[:4])
        tmp_enc_2 = nn.Sequential(*tmp_layers[4:11])
        tmp_enc_3 = nn.Sequential(*tmp_layers[11:18])
        tmp_enc_4 = nn.Sequential(*tmp_layers[18:31])

        self.enc_1.load_state_dict(tmp_enc_1.state_dict())
        self.enc_2.load_state_dict(tmp_enc_2.state_dict())
        self.enc_3.load_state_dict(tmp_enc_3.state_dict())
        self.enc_4.load_state_dict(tmp_enc_4.state_dict())

        logger.info(f'Load ckpt of [Encoder] from : {encoder_path}.')

        # 载入 decoder 权重续训
        decoder_path = ckpt_dict.get('decoder_path',None)
        if decoder_path is not None and os.path.isfile(decoder_path):
            decoder_state_dict = torch.load(decoder_path,map_location='cpu')
            self.decoder.load_state_dict(decoder_state_dict)
            logger.info(f'Load ckpt of [Decoder] from : {decoder_path}.')
            

    # @property
    # def trainable_params(self):
    #     return list(self.decoder.parameters()) + \
    #            list(self.fc_encoder.parameters()) + \
    #            list(self.fc_decoder.parameters())

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]
        
    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        # assert (target.requires_grad is False)
        return F.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return F.mse_loss(input_mean, target_mean) + \
               F.mse_loss(input_std, target_std)

    def calc_domain_KL_loss(self,norm_style_stats,eps=1e-8):
        bz,d = norm_style_stats.shape 
        c = d // 2 
        
        mu = norm_style_stats[:,:c]
        sigma = norm_style_stats[:,c:] + eps 

        var = sigma.pow(2)
        logvar = var.log() 

        # KL per dimension:
        # 0.5 * (mu^2 + sigma^2 - log(sigma^2) - 1)
        kl = .5 * (mu.pow(2) + var - logvar - 1.)

        return kl.mean()
     
    def calc_domain_Rec_loss(self,ori_style_stats,rec_style_states):
        return F.mse_loss(ori_style_stats,rec_style_states,reduction='sum')

    def forward(self,content, style, alpha=1.0):
        assert 0.0 <= alpha <= 1.0 

        if self.training: 
            content_feat = self.encode(content)
            style_feats = self.encode_with_intermediate(style)

            # 1. train Domain-VAE
            ori_style_stats = calc_feat_mean_std(style_feats[-1]) # [B,1024]
            norm_style_stats = self.fc_encoder(ori_style_stats)   # [B,1024]
            rec_style_stats = self.fc_decoder(norm_style_stats[:,:512])

            # 2. Apply AdaIN
            t = adain(content_feat, rec_style_stats)
            t = alpha * t + (1 - alpha) * content_feat
            g_t = self.decoder(t)   

            # 3. calc Loss 

            # 3.1.1 calc loss about AdaIN
            g_t_feats = self.encode_with_intermediate(g_t)

            loss_c = self.calc_content_loss(g_t_feats[-1], t)
            loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
            for i in range(1, 4):
                loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])

            # 3.1.2 calc loss about Domain-VAE
            loss_KL = self.calc_domain_KL_loss(norm_style_stats)
            loss_Rec = self.calc_domain_Rec_loss(ori_style_stats,rec_style_stats)

            loss_dict = {
                'loss_c'  : loss_c,
                'loss_s'  : loss_s,
                'loss_KL' : loss_KL,
                'loss_Rec' : loss_Rec,
            }

            return loss_dict,g_t

        else:
            content_feat = self.encode(content)
            rec_style_stats = self.fc_decoder(style)
        
            t = adain(content_feat, style_feats[-1])
            t = alpha * t + (1 - alpha) * content_feat
            g_t = self.decoder(t)   

            return g_t