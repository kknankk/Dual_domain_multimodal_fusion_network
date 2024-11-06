# -*- coding: utf-8 -*-

import os
import numpy as np
from PIL import Image
import pandas as pd 

import torch
from torch.utils.data import Dataset
# import 
import glob
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import os
import numpy as np
import scipy.signal as sig
import numpy as np
import pandas as pd
#b,c,h,w
#b,c d
import h5py
import torch # to set manual seed
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import wfdb # wfdb?
# mport torch 
# torch.set_printoptions(threshold=np.inf)

# Adjust ECGs to match Code-15 standard
def adjust_sig(signal):
    # MIMIC IV signals are 10s at 500Hz, 5k total samples
    # we seek 10s @ 400Hz, centered, padded to 4096.
    # so resample to 4k samples
    # then pad with 48 0's on both sides of axis=0
    new_sig = sig.resample(signal, 4000, axis=0)
    #TODO: change （1，1）to（48，48）
    new_sig = np.pad(new_sig, ((1,1),(0,0)), mode='constant')
    new_sig = new_sig.astype(np.float32) # store as float32, else OOM
    return new_sig
    
def generate_file_path(base_path, split):
    # 使用 f-string 生成路径
    return f"{base_path}{split}.csv"

class MIMICECG(Dataset):
    def __init__(self,  fullinfo_file,transform=None, split='train',module='fusion'):#delete args in params
        # self.data_dir = args.cxr_data_dir
        # cxr_data_dir='/data/ke/data/MIMIC/physionet.org/files/mimic-cxr-jpg/2.0.0'
        # fullinfo_file='/data/ke/MIMIC_subset/PA_subset/with_nonan_label_PA_'
        # self.args = args
    #     self.CLASSES  = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    #    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    #    'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
    #    'Pneumonia', 'Pneumothorax', 'Support Devices']\
        self.split=split
        self.module = module
        self.fullinfo_path=fullinfo_file
        file_path = generate_file_path(fullinfo_file, self.split)
        # print(file_path)#/data/ke/MIMIC_subset/PA_subset/with_nonan_label_PA_train.csv
        self.ecg_dir='/data/ke/data/physionet.org/files/mimic-iv-ecg/1.0'

        # self.filenames_to_path = {path.split('/')[-1].split('.')[0]: path for path in paths}
        # self.base_path='/data/ke/MIMIC_subset/PA_subset/with_nonan_label_PA_'
        with open(file_path, "r")as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]
        self.CLASSES=self._listfile_header.strip().split(',')[6:13]
        self._data = self._data[1:]

        self._data = [line.split(',') for line in self._data]

        # label_indices = {label: idx + 6 for idx, label in enumerate(self.CLASSES)}

        self.data_map = {}

        self.data_map = {
            mas[1]: {
                'labels': [1 if label == '1.0' else 0 for label in mas[6:13]],
                'hadm_id': float(mas[1]),
                'first_ecg_path': mas[4],
                # 'dicom_id': float(mas[14]),
                }
                for mas in self._data
        }
        print(len(self.data_map))#train 10632;test/val:1653
        self.names = list(self.data_map.keys())
        # print(self.data_map)
        self.transform = transform

    def __getitem__(self, index):
        # print(f'get_item index {index}')
        if isinstance(index, int):
            index1 = self.names[index]
            # print(index)#is hadm_id
            y = self.data_map[index1]['labels']
            # print(y)
            study_id_path = self.data_map[index1]['first_ecg_path']
            rec_path=f'{self.ecg_dir}/{study_id_path}'
            # print(f'get item path {rec_path}')
            rd_record = wfdb.rdrecord(rec_path)
            
            # fig=wfdb.plot_wfdb(record=rd_record, figsize=(124,18), title='Study 41420867 example', ecg_grids='all', return_fig=True)
            # fig.savefig(f'ecg_record_{index}.png')  # 保存文件名可以根据需要调整
            # print(f'success save ')
        # 关闭图像以释放内存

            # print(rec_path)
            signal = rd_record.p_signal # 5k x 12 signal  #not none
            # print(f'after rd_record {signal}')
            # if np.isnan(signal).any():
            #     # 可以选择抛出异常或返回 None
            #     # raise ValueError(f'Signal contains NaN values for index: {index}')
            #     return None

            if np.isnan(signal).any():
                return None  # 返回 None 或者其他标记值，表示该实例无效
                # return self.__getitem__((index + 1) % len(self.names))  # 跳过 NaN 数据，选择下一个样本
   

            signal=adjust_sig(signal)#4096,12
            signal = torch.tensor(signal, dtype=torch.float32)
            signal=signal.permute(1,0)
            
            # print(f'after adjust signal{signal}')
            # signal = np.transpose(signal, (1, 0))
            # signal2=0
            # print(signal.shape)

#normalize
            if self.split=='train':
                # print(f'get item signal {signal.shape}')
                # mean=torch.tensor([0.010844607, 0.0062680915, -0.0039852564, -0.008372346, 0.0008754317, 0.00707275, -0.008885117, -0.001465515, 0.00030259503, 0.00712353, 0.008747238, 0.010062958])
                # std=torch.tensor([0.07435964, 0.0697202, 0.07066378, 0.062104683, 0.05933149, 0.06400871, 0.08684388, 0.11842892, 0.12385771, 0.11148283, 0.09947806, 0.085521184])

                mean=torch.tensor([0.022676835, 0.013688515, -0.007821267, -0.017821848, 0.0023585667, 0.014657727, -0.018340515, -0.0038577598, -0.00035443218, 0.014264569, 0.020447042, 0.019408429])
                std=torch.tensor([0.15746197, 0.15279943, 0.16141628, 0.13161984, 0.1358389, 0.1403546, 0.19144094, 0.2620128, 0.27269644, 0.24160655, 0.21423854, 0.18789752])

                signal=(signal-mean.view(-1,1)) / std.view(-1,1)
                # print(f'after norm signal {signal}')

                # with open ('normalized get item.txt','w') as f:
                #     f.write(str(signal))
                #     print(f'finish writting in txt')
                    # break
                # print(f'get item normalized signal {signal}')
            if self.split=='val':
                # print(f'get item signal {signal.shape}')
                # mean=torch.tensor([0.010844607, 0.0062680915, -0.0039852564, -0.008372346, 0.0008754317, 0.00707275, -0.008885117, -0.001465515, 0.00030259503, 0.00712353, 0.008747238, 0.010062958])
                # std=torch.tensor([0.07435964, 0.0697202, 0.07066378, 0.062104683, 0.05933149, 0.06400871, 0.08684388, 0.11842892, 0.12385771, 0.11148283, 0.09947806, 0.085521184])

                mean=torch.tensor([0.022080505, 0.012505924, -0.008371157, -0.016918872, 0.0015192, 0.014522009, -0.018306602, -0.003076632, 0.000574659, 0.0146055035, 0.01795028, 0.02059472])
                std=torch.tensor([0.16260016, 0.1545154, 0.16712894, 0.13399345, 0.13874556, 0.1459843, 0.19639233, 0.25812393, 0.2829126, 0.24955757, 0.22786401, 0.18867913])
                signal=(signal-mean.view(-1,1)) / std.view(-1,1)




            if self.transform is not None:
                # signal = torch.tensor(signal, dtype=torch.float32)
                signal = self.transform(signal)
                

            if self.module=='fusion':

                return {index:(signal, y)}
            else:
                # print(f'model!=fusion signal {signal}')
                return signal,y

    def __len__(self):
        return len(self.data_map)
    
 

def get_ECG_datasets(module='unimodal'):#delete args in params
    # train_transforms,val_transforms, test_transforms = get_transforms()#delete args in params


    fullinfo_file='/data/ke/MIMIC_subset/PA_subset/with_nonan_label_PA_'
    # dataset_train = MIMICECG(fullinfo_file, split='train',transform=transforms.Compose(train_transforms),module=module)#delete args in params
    # dataset_validate = MIMICECG(fullinfo_file, split='val',transform=transforms.Compose(val_transforms),module=module)#delete args in params
    # dataset_test = MIMICECG(fullinfo_file, split='test',transform=transforms.Compose(test_transforms),module=module)#delete args in params
    dataset_train = MIMICECG(fullinfo_file, split='train',module=module)#delete args in params
    dataset_validate = MIMICECG(fullinfo_file, split='val',module=module)#delete args in params
    dataset_test = MIMICECG(fullinfo_file, split='test',module=module)#delete args in params

    return dataset_train, dataset_validate, dataset_test


def get_data_loader(batch_size):
    train_ds, val_ds, test_ds = get_ECG_datasets()
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=16)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False, collate_fn=my_collate, pin_memory=True, num_workers=16)

    return train_dl, val_dl

def my_collate(batch):
    # 将样本的输入数据（x）堆叠成一个批次
    #TODO: initial:
    batch = [item for item in batch if item is not None]

    # for item in batch:
    #     print(item[0].shape)
    #     batch.append(item) 


    x = np.stack([item[0] for item in batch])

# 
    
    # 将目标标签（y）转换为NumPy数组
    targets = np.array([item[1] for item in batch])
    
    # 返回输入数据和对应的目标标签
    return [x, targets]


# A,B,C=get_ECG_datasets()

# # label = A.__getitem__(index=5)
# # # print(f"Signal: {signal}")
# # print(f"Labels: {label}")

ehr_train,ehr_test,ehr_val=get_ECG_datasets()
# a=ehr_train.__getitem__(index=5)
# print(a) #
# {'22928046': (array([[0., 0., 0., ..., 0., 0., 0.],
#        [0., 0., 0., ..., 0., 0., 0.],
#        [0., 0., 0., ..., 0., 0., 0.],
#        ...,
#        [0., 0., 0., ..., 0., 0., 0.],
#        [0., 0., 0., ..., 0., 0., 0.],
#        [0., 0., 0., ..., 0., 0., 0.]], dtype=float32), [0, 0, 0, 0, 1, 0, 0])}
# # print(f"Signal: {signal}")
# # print(f'y: {y}')
# # print(f"Signal2: {signal2}")
# for a,b in ehr_train:
#     print(a,b) #a is none
#     continue
