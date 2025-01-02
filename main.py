
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

# deeper_frequency_fusion_TWO
from train.deeper_fusion_trainer_mod import deeper_fusion_trainer_mod








from model.DDMF_Net import DDMF_Net


# from model.fourinput_model import FSRU


from model.dr_fuse import DrFuseModel
from model.med_fuse import medfuse
from model.mmtm import mmtm_med
from model.modified_medfuse import mod_medfuse

# from model.modified_Units import mod_UniTS

from argument import args_parser
parser = args_parser()
# add more arguments here ...
args = parser.parse_args()





# train_loader,val_loader=get_data_loader(batch_size=16)





fusion_train_dl,fusion_val_dl=get_ecgcxr_data_loader(batch_size=args.batch_size)




if args.fusion_model=='DDMF_Net':
    fusion_model=DDMF_Net()




elif args.fusion_model=='drfuse':
    fusion_model=DrFuseModel(hidden_size=args.hidden_size,
                                 num_classes=4,
                                 ehr_dropout=0.3,
                                 ehr_n_head=args.ehr_n_head,
                                 ehr_n_layers=args.ehr_n_layers)
# elif args.fusion_model=='medfuse':
#     fusion_model=medfuse()    



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




