#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File     :     hubconf.py
@Time     :     2025/11/24 14:27:49
@Author   :     Louis Swift
@Desc     :     
    


        import torch 
        from omegaconf import OmegaConf

        config = OmegaConf.load('config\\model\\DSP.yaml')
        torch.hub.load(
            repo_or_dir='../DSP',
            model=config.name,
            source='local',
            ckpt_dict = config
        )

        # NOTE : you have to download the relavent ckpt and place it in the proper location.

'''

import torch
from model.net import DSPNet

def DSP(pretrained=True, **kwargs):
    """
    pretrained (bool): load pretrained weights into model
    """

    # ckpt_path = kwargs.get('ckpt_path',None)
    # ckpt_dict = {}
    # if ckpt_path is None:
    #     ckpt_dict['ckpt_path'] = 'https://github.com/Kiumb1223/DomainStylePerturbation-Unofficial/releases/download/trained-ckpt/model_iter_160000.pth'

    model = DSPNet(**kwargs)
    if pretrained:
        ckpt_path = kwargs.get('ckpt_path',None)
        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path, map_location='cpu')
        else:
            state_dict = torch.hub.load_state_dict_from_url(
                url='https://github.com/Kiumb1223/DomainStylePerturbation-Unofficial/releases/download/trained-ckpt/model_iter_160000.pth',
                map_location='cpu',
                check_hash=True
            )

        model.load_state_dict(state_dict)
        
    return model