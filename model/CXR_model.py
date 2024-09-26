
import torch.nn as nn
import torchvision
import torch
import numpy as np

from torch.nn.functional import kl_div, softmax, log_softmax
# from .loss import RankingLoss, CosineLoss
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys
import os
class CXRModels(nn.Module):

    def __init__(self,):
	
        super(CXRModels, self).__init__()
        # self.args = args

        self.vision_backbone = getattr(torchvision.models, 'resnet34')(pretrained=True) #get attribute
        classifiers = [ 'classifier', 'fc']
        for classifier in classifiers:
            cls_layer = getattr(self.vision_backbone, classifier, None)
            if cls_layer is None:
                continue
            d_visual = cls_layer.in_features #找到输入特征数
            setattr(self.vision_backbone, classifier, nn.Identity(d_visual)) #set attribute
            break
        # self.bce_loss = torch.nn.BCELoss(size_average=True)
        self.classifier = nn.Sequential(nn.Linear(d_visual, 7))
        self.feats_dim = d_visual
       

    def forward(self, x, labels=None, n_crops=0, bs=16):
        # lossvalue_bce = torch.zeros(1)
        # print(f'input cxr size {x.shape}') #[batch_size,channel=3,224,224]
        visual_feats = self.vision_backbone(x)
        preds = self.classifier(visual_feats)

        # preds = torch.sigmoid(preds)


        return preds


  
