
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
# from dataset.ECG_dataset import get_ECG_datasets,get_data_loader
from dataset.update_ECGdataset import get_ECG_datasets,get_data_loader
from dataset.cxr_dataset import get_cxr_datasets,get_cxrdata_loader
from dataset.fusion_dataset import load_cxr_ecg_ds,get_ecgcxr_data_loader
import numpy as np
# from model.ECG_model import LSTM
from train.general_trainer import G_trainer
from train.fusion_trainer import Fusion_trainer
from train.drfuse_trainer import Dr_trainer
from train.medfuse_trainer import medfuse_trainer
from train.mod_medfuse_trainer import mod_medfuse_trainer

from train.deeper_fusion_trainer import deeper_fusion_trainer
from train.fourinput_se_a_j_trainer import fourinput_saj_trainer
from train.TSRNet_trainer import TSRNet_trainer
from model.ECG_model import Spect_CNN,spectrogram_model,CustomResNet18
from model.CXR_model import wavevit_s
from model.gfnet import GFNet
#先用fourinput_model
# from model.fusion_model import FSRU
# from model.med_fuse import CXRModels
# from model.ECG_model import ResNet1d
from model.fourinput_model import CXRModels,ResNet1d
#TODO:调试senet时先用
#调试senet时，先用fourinput_senet中的FSRU
from model.fourinput_senet import FSRU
# from model.fourinput_model import FSRU
from model.fourinput_saj import FSRU_A

from model.ECG_fusion import UniTS
from model.dr_fuse import DrFuseModel
from model.med_fuse import medfuse
from model.modified_medfuse import mod_medfuse
from model.TSRNet import TSRNet
# from model.modified_Units import mod_UniTS

from argument import args_parser
parser = args_parser()
# add more arguments here ...
args = parser.parse_args()

import random
seed = 42
random.seed(seed)  # 设置 Python 随机种子
np.random.seed(seed)  # 设置 NumPy 随机种子
torch.manual_seed(seed)  # 设置 PyTorch 随机种子
torch.cuda.manual_seed(seed)  # 设置当前 GPU 随机种子
torch.cuda.manual_seed_all(seed)  # 设置所有 GPU 随机种子




# train_loader,val_loader=get_data_loader(batch_size=16)
#-----先注释ecg数据
ehr_train,ehr_test,ehr_val=get_ECG_datasets(args)

train_ecg_dl, val_ecg_dl = get_data_loader(batch_size=args.batch_size)
#-----先注释ecg数据

train_ds,val_ds,test_ds=get_cxr_datasets()
train_cxr_dl,val_cxr_dl,test_cxr_dl=get_cxrdata_loader(batch_size=args.batch_size)

#----------先注释fusion数据
fusion_train_dl,fusion_val_dl=get_ecgcxr_data_loader(batch_size=args.batch_size)
#----------先注释fusion数据

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
# elif args.ecg_model == 'ECGModel':
#     ecg_model = ECGModel()
elif args.ecg_model == 'spect':
    ecg_model = Spect_CNN()
elif args.ecg_model == 'resnet18':
    ecg_model = CustomResNet18()
elif args.ecg_model == 'units':
    ecg_model = UniTS()
elif args.ecg_model == 'tsrnet':
    ecg_model = TSRNet(enc_in=12)

# else:
#     raise ValueError(f"Unknown ECG model: {args.ecg_model}")

if args.cxr_model == 'wavevit_s':
    cxr_model = wavevit_s()
elif args.cxr_model == 'CXRModels':
    cxr_model = CXRModels()
elif args.cxr_model == 'gfnet':
    cxr_model = GFNet()
# else:
#     raise ValueError(f"Unknown CXR model: {args.cxr_model}")

if args.fusion_model=='FSRU':
    fusion_model=FSRU()
elif args.fusion_model=='FSRU_SAJ':
    fusion_model=FSRU_A()


elif args.fusion_model=='drfuse':
    fusion_model=DrFuseModel(hidden_size=args.hidden_size,
                                 num_classes=4,
                                 ehr_dropout=0.3,
                                 ehr_n_head=args.ehr_n_head,
                                 ehr_n_layers=args.ehr_n_layers)
# elif args.fusion_model=='medfuse':
#     fusion_model=medfuse()    


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

if args.fusion_type=='deeper_frequency_fusion' :
    print(f'--------start fusion training-------------')
    trainer=deeper_fusion_trainer(fusion_train_dl, fusion_val_dl,args,fusion_model)
    trainer.train()

elif args.fusion_type=='fourinput_saj_fusion' :
    print(f'--------start fusion training-------------')
    trainer=fourinput_saj_trainer(fusion_train_dl, fusion_val_dl,args,fusion_model)
    trainer.train()

elif args.fusion_type=='fusion':
    print(f'--------start fusion training-------------')
    trainer=Fusion_trainer(fusion_train_dl, fusion_val_dl,args,ecg_model,cxr_model)
    trainer.train()

elif args.fusion_type=='ecg_fusion' :
    print(f'--------start ecg_fusion training-------------')
    trainer=TSRNet_trainer(train_ecg_dl, val_ecg_dl,args,ecg_model)
    trainer.train()

elif args.fusion_type=='drfuse':
    print(f'--------start fusion training-------------')
    trainer=Dr_trainer(fusion_train_dl, fusion_val_dl,args,fusion_model)
    trainer.train()

#先注掉原版medfuse
elif args.fusion_type=='mod_medfuse':
    print(f'--------start fusion training-------------')
    trainer=mod_medfuse_trainer(fusion_train_dl, fusion_val_dl,args)
    trainer.train()

elif args.fusion_type=='medfuse':
    print(f'--------start fusion training-------------')
    trainer=medfuse_trainer(fusion_train_dl, fusion_val_dl,args)
    trainer.train()


