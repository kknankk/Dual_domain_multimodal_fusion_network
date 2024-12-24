
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os
import numpy as np

import random
seed = 42
random.seed(seed)  
np.random.seed(seed)  
torch.manual_seed(seed) 
torch.cuda.manual_seed(seed)  
torch.cuda.manual_seed_all(seed)  
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark = False

sys.path.append(os.path.abspath('/home/ke/MIMIC_subset/MIMIC_subset'))
# print(a)
# from dataset.ECG_dataset import get_ECG_datasets,get_data_loader
from dataset.update_ECGdataset import get_ECG_datasets,get_data_loader
from dataset.cxr_dataset import get_cxr_datasets,get_cxrdata_loader

# from dataset.fusion_dataset import load_cxr_ecg_ds,get_ecgcxr_data_loader
from dataset.fusion_dataset_ONE import load_cxr_ecg_ds,get_ecgcxr_data_loader

import numpy as np
# from model.ECG_model import LSTM
from train.general_trainer import G_trainer
from train.fusion_trainer import Fusion_trainer
from train.drfuse_trainer import Dr_trainer
from train.medfuse_trainer import medfuse_trainer
from train.mod_medfuse_trainer import mod_medfuse_trainer

from train.deeper_fusion_trainer import deeper_fusion_trainer
from train.deeper_fusion_trainer_TWO import deeper_fusion_trainer_TWO
# deeper_frequency_fusion_TWO
from train.deeper_fusion_trainer_mod import deeper_fusion_trainer_mod
from train.deeper_fusion_trainer_se_ba import deeper_fusion_trainer_se_ba
from train.fourinput_se_a_j_trainer import fourinput_saj_trainer
from train.TSRNet_trainer import TSRNet_trainer
from model.ECG_model import Spect_CNN,spectrogram_model,CustomResNet18
from model.CXR_model import wavevit_s
from model.gfnet import GFNet

# from model.fusion_model import FSRU
from model.frsu_bottleneck_attention import FSRU_BA
# from model.med_fuse import CXRModels
# from model.ECG_model import ResNet1d
from model.fourinput_model import CXRModels,ResNet1d



from model.DDMF_Net import DDMF_Net


# from model.fourinput_model import FSRU
from model.fourinput_saj import FSRU_A
from model.fourinput_senet_mod import FSRU_mod
from model.fusion_model_new import FSRU_NEW
from model.ECG_fusion import UniTS
from model.dr_fuse import DrFuseModel
from model.med_fuse import medfuse
from model.mmtm import mmtm_med
from model.modified_medfuse import mod_medfuse
from model.TSRNet import TSRNet
# from model.modified_Units import mod_UniTS

from argument import args_parser
parser = args_parser()
# add more arguments here ...
args = parser.parse_args()





# train_loader,val_loader=get_data_loader(batch_size=16)

ehr_train,ehr_val=get_ECG_datasets(args)

train_ecg_dl, val_ecg_dl = get_data_loader(batch_size=args.batch_size)


train_ds,val_ds=get_cxr_datasets()
train_cxr_dl,val_cxr_dl=get_cxrdata_loader(batch_size=args.batch_size)


fusion_train_dl,fusion_val_dl=get_ecgcxr_data_loader(batch_size=args.batch_size)


# with open('path/to/result.txt') as result_file:
#     result_file.write(#TODO)




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


elif args.fusion_model=='DDMF_Net':
    fusion_model=DDMF_Net()




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

# if args.fusion_type=='deeper_frequency_fusion' :
#     print(f'--------start fusion training-------------')
#     trainer=deeper_fusion_trainer(fusion_train_dl, fusion_val_dl,args,fusion_model)
#     trainer.train()
# elif args.fusion_type=='deeper_frequency_fusion_test' :
#     print(f'--------start fusion training-------------')
#     trainer=deeper_fusion_trainer(fusion_train_dl, fusion_val_dl,args,fusion_model)
#     trainer.test()


elif args.fusion_type=='deeper_frequency_fusion_mod' :
    print(f'--------start fusion training-------------')
    trainer=deeper_fusion_trainer_mod(fusion_train_dl, fusion_val_dl,args,fusion_model)
    trainer.train()
elif args.fusion_type=='deeper_frequency_fusion_mod_test' :
    print(f'--------start fusion training-------------')
    trainer=deeper_fusion_trainer_mod(fusion_train_dl, fusion_val_dl,args,fusion_model)
    trainer.test()

elif args.fusion_type=='drfuse':
    print(f'--------start fusion training-------------')
    trainer=Dr_trainer(fusion_train_dl, fusion_val_dl,args,fusion_model)
    trainer.train()
elif args.fusion_type=='drfuse_test':
    print(f'--------start fusion training-------------')
    trainer=Dr_trainer(fusion_train_dl, fusion_val_dl,args,fusion_model)
    trainer.test()


elif args.fusion_type=='mod_medfuse':
    print(f'--------start fusion training-------------')
    trainer=mod_medfuse_trainer(fusion_train_dl, fusion_val_dl,args)
    trainer.train()


elif args.fusion_type=='medfuse':
    print(f'--------start fusion training-------------')
    trainer=medfuse_trainer(fusion_train_dl, fusion_val_dl,args)
    trainer.train()
elif args.fusion_type=='medfuse_test':
    print(f'--------start fusion training-------------')
    trainer=medfuse_trainer(fusion_train_dl, fusion_val_dl,args)
    trainer.test()




