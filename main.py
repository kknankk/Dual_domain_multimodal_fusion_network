
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os

# 将项目根目录添加到 Python 路径
sys.path.append(os.path.abspath('/data/ke/MIMIC_subset'))
# print(a)
from dataset.ECG_dataset import get_ECG_datasets,get_data_loader
from dataset.cxr_dataset import get_cxr_datasets,get_cxrdata_loader
import numpy as np
from model.ECG_model import LSTM
from train.general_trainer import G_trainer
from model.ECG_model import LSTM,Spect_CNN,ECGModel
from model.CXR_model import CXRModels

from argument import args_parser
parser = args_parser()
# add more arguments here ...
args = parser.parse_args()

# train_loader,val_loader=get_data_loader(batch_size=16)
ehr_train,ehr_test,ehr_val=get_ECG_datasets()
train_ecg_dl, val_ecg_dl = get_data_loader(batch_size=16)

train_ds,val_ds,test_ds=get_cxr_datasets()
train_cxr_dl,val_cxr_dl,test_cxr_dl=get_cxrdata_loader(batch_size=16)

ecg_model=ECGModel()
cxr_model=CXRModels()
# cxr_train_dl,cxr_val_dl,cxr_test_dl=

# ecg_train_dl,ecg_val_dl,ecg_test_dl=

# fuse_train_dl,fuse_val_dl,fuse_test_dl=


# with open('path/to/result.txt') as result_file:
#     result_file.write(#TODO)

if args.fusion_type=='ecg':
    print(f'--------start ecg training-------------')
    trainer=G_trainer(train_ecg_dl, val_ecg_dl,args,ecg_model)
    trainer.train()

if args.fusion_type=='cxr':
    print(f'--------start cxr training-------------')
    trainer=G_trainer(train_cxr_dl, val_cxr_dl,args,cxr_model)
    trainer.train()
# elif args.fusion_type=='ecg':
#     trainer=Trainer(#TODO)

# else:
#     args.fusion_type=='fusion':
#     trainer=Trainer(#TODO)


