
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
sys.path.append(os.path.abspath('/home/ke/MIMIC_subset/MIMIC_subset'))
# print(a)
from dataset.ECG_dataset import get_ECG_datasets,get_data_loader
from dataset.cxr_dataset import get_cxr_datasets,get_cxrdata_loader
from dataset.fusion_dataset import load_cxr_ecg_ds,get_ecgcxr_data_loader
import numpy as np
from model.ECG_model import LSTM
from train.general_trainer import G_trainer
from train.fusion_trainer import Fusion_trainer
from model.ECG_model import LSTM,Spect_CNN,ECGModel,ResNet1d,spectrogram_model
from model.CXR_model import CXRModels,wavevit_s

from argument import args_parser
parser = args_parser()
# add more arguments here ...
args = parser.parse_args()

# train_loader,val_loader=get_data_loader(batch_size=16)
ehr_train,ehr_test,ehr_val=get_ECG_datasets()
train_ecg_dl, val_ecg_dl = get_data_loader(batch_size=args.batch_size)

train_ds,val_ds,test_ds=get_cxr_datasets()
train_cxr_dl,val_cxr_dl,test_cxr_dl=get_cxrdata_loader(batch_size=args.batch_size)

fusion_train_dl,fusion_val_dl=get_ecgcxr_data_loader(batch_size=args.batch_size)


# with open('path/to/result.txt') as result_file:
#     result_file.write(#TODO)



# # 根据传入的模型名称选择模型
# if args.model == 'ResNet1d':
#     ecg_model = ResNet1d()
# elif args.model == 'spectrogram':
#     ecg_model = spectrogram_model()
# elif args.model == 'ECGModel':
#     ecg_model = ECGModel()
# elif args.model == 'wavevit_s':
#     cxr_model = wavevit_s()
# elif args.model == 'CXRModels':
#     cxr_model = CXRModels()
# else:
#     raise ValueError(f"Unknown model: {args.model}")
if args.ecg_model == 'ResNet1d':
    ecg_model = ResNet1d()
elif args.ecg_model == 'spectrogram':
    ecg_model = spectrogram_model()
elif args.ecg_model == 'ECGModel':
    ecg_model = ECGModel()
else:
    raise ValueError(f"Unknown ECG model: {args.ecg_model}")

if args.cxr_model == 'wavevit_s':
    cxr_model = wavevit_s()
elif args.cxr_model == 'CXRModels':
    cxr_model = CXRModels()
else:
    raise ValueError(f"Unknown CXR model: {args.cxr_model}")



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

elif args.fusion_type=='fusion':
    print(f'--------start fusion training-------------')
    trainer=Fusion_trainer(fusion_train_dl, fusion_val_dl,args,ecg_model,cxr_model)
    trainer.train()

