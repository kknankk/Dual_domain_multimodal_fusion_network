
# import sys
# import os
# sys.path.append(os.path.abspath('/home/mimic/MIMIC_subset/MIMIC_subset'))
# from torch.utils.data import Dataset
# from dataset.ECG_dataset import get_ECG_datasets
# from dataset.cxr_dataset import get_cxr_datasets
# from torch.utils.data import DataLoader
# import numpy as np
# from torch.utils.data import Dataset
# import torch

# def generate_file_path(base_path, split):
#     # 使用 f-string 生成路径
#     return f"{base_path}{split}.csv"

# class MIMIC_CXR_EHR(Dataset):
#     def __init__(self,ehr_ds,cxr_ds,split='train'):
#         fullinfo_file='/home/mimic/MIMIC_subset/MIMIC_subset/PA_subset/with_nonan_label_PA_'
#         file_path = generate_file_path(fullinfo_file, split)
#         self.ecg_dir='/home/mimic/MIMIC_subset/MIMIC_subset/ECG_data'

#         with open(file_path, "r")as lfile:
#             self._data = lfile.readlines()
#         self._listfile_header = self._data[0]
#         self._data = self._data[1:]
#         self.CLASSES=self._listfile_header.strip().split(',')[6:13]
#         self._data = [line.split(',') for line in self._data]

#         # label_indices = {label: idx + 6 for idx, label in enumerate(self.CLASSES)}

#         self.data_map = {}

#         self.data_map = {
#             mas[1]: {
#                 'labels': [1 if label == '1.0' else 0 for label in mas[6:13]],
#                 'hadm_id': float(mas[1]),
#                 'first_ecg_path': mas[4],
#                 # 'dicom_id': float(mas[14]),
#                 }
#                 for mas in self._data
#         }
#         # print(self.data_map)
#         # print(len(self.data_map))#train 10632;test/val:1653
#         self.names = list(self.data_map.keys())
#         # print(self.names[2])#21566603
#         # print(self.data_map)# right
#         # self.transform = transform
#         # self.ecg_dir='/data/ke/data/physionet.org/files/mimic-iv-ecg/1.0'
#         self.ehr_ds=ehr_ds
#         # print(f'self.ehr_ds[5]: {self.ehr_ds[5]}') #index is natural number rather than hadm_id
#         # data=
#         # for signal,y in ehr_ds:
#         #     print(signal,y)#正常的
#         self.cxr_ds=cxr_ds
#         self.split=split

#     def __getitem__(self, index):
#         # y = self.data_map[index]['labels']
#         individual_hadm_id = self.names[index]
#         # prin(f'index: {index}')
#         # breakpoint()
#         # print(f'self.ehr_ds{index}: {self.ehr_ds[index]}')#index is natural number rather than hadm_id
#         # ehr_data,y=self.ehr_ds[self.data_map[index]]
#         ehr_data,y=self.ehr_ds[index].get(individual_hadm_id)

#         # cxr_data=self.cxr_ds[self.data_map[index]]
#         cxr_data=self.cxr_ds[index][individual_hadm_id]
#         return ehr_data,cxr_data,y

#     def __len__(self):
#         return len(self.data_map)




# def load_cxr_ecg(ecg_train_ds, ecg_val_ds, cxr_train_ds, cxr_val_ds, ecg_test_ds, cxr_test_ds):

#     ecg_train_ds, ecg_val_ds, ecg_test_ds = get_ECG_datasets(module='fusion')

#     cxr_train_ds, cxr_val_ds, cxr_test_ds = get_cxr_datasets(module='fusion')

#     train_ds = MIMIC_CXR_EHR(ecg_train_ds,cxr_train_ds,split='train')
#     val_ds = MIMIC_CXR_EHR(ecg_val_ds,cxr_val_ds,split='val')
#     test_ds = MIMIC_CXR_EHR(ecg_test_ds,cxr_test_ds,split='test')

#     train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=16, drop_last=True)
#     val_dl = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=my_collate, pin_memory=True, num_workers=16, drop_last=False)
#     test_dl = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=my_collate, pin_memory=True, num_workers=16, drop_last=False)

#     return train_dl, val_dl, test_dl

# # def my_collate(batch):
# #     # 将样本的输入数据（x）堆叠成一个批次
# #     ehr_data = np.stack([item[0] for item in batch])
    
# #     # 将目标标签（y）转换为NumPy数组
# #     cxr_data = np.array([item[1] for item in batch])
# #     target=
# #     # 返回输入数据和对应的目标标签
# #     return [ehr_data,cxr_data,targets]

# def my_collate(batch):
#     ecg_data = [item[0] for item in batch]
#     # pairs = [False if item[1] is None else True for item in batch]
#     img = torch.stack([item[1] for item in batch])
#     # x, seq_length = pad_zeros(x)
#     targets = np.array([item[2] for item in batch])
#     # targets_cxr = torch.stack([torch.zeros(14) if item[3] is None else item[3] for item in batch])
#     return [ecg_data,img,targets]


# #-----访问fusion_dataloader测试
# ehr_train,ehr_test,ehr_val=get_ECG_datasets(module='fusion')
# cxr_train,cxr_test,cxr_val=get_cxr_datasets(module='fusion')

# train_dl, val_dl,test_dl = load_cxr_ecg(ehr_train,ehr_val,cxr_train,cxr_val,ehr_test,cxr_test)

# # 访问train_dl中的数据
# for batch_idx, batch in enumerate(train_dl):
#     # batch 是从 collate_fn 处理后的结果
#     signal,cxr_data, label = batch  # 这里假设 my_collate 函数返回(signal, label)
    
#     print(f"Batch {batch_idx}:")
#     print(f"Signal shape: {len(signal)}")  # 打印信号的形状
#     print(f'cxr shape: {cxr_data.shape}')
#     print(f"Labels: {label}")  # 打印标签
    
    # 你可以在这里对数据进行进一步处理或训练模型
    # break  # 如果你只想查看第一个 batch 的数据，可以使用 break




#-----------------test part
# import logging

# # 设置logger
# logger = logging.getLogger('MIMIC_logger')
# logger.setLevel(logging.INFO)

# # 创建文件处理器来保存日志
# file_handler = logging.FileHandler('MIMIC_output.log')
# file_handler.setLevel(logging.INFO)


# ehr_train,ehr_test,ehr_val=get_ECG_datasets(module='fusion')
# for a,b in ehr_train:
#     print(a,b)
# for index,(signal,y) in enumerate(ehr_train):
    # print(signal,y)
# print(ehr_train)
# cxr_train,cxr_test,cxr_val=get_cxr_datasets(module='fusion')
# train_dl, val_dl, test_dl = load_cxr_ehr(ecg_train_ds, ecg_val_ds, cxr_train_ds, cxr_val_ds, ecg_test_ds, cxr_test_ds)
# train_ds = MIMIC_CXR_EHR(ehr_train,cxr_train,split='train')
# for i,(ehr_data,cxr_data, y) in enumerate(train_ds):
#     print(f'ehr_data: {ehr_data}')
#     print(f'cxr_data: {cxr_data}')
#     print(f'y: {y}')
#----------above all right
# ehr_data,cxr_data, y=train_ds.__getitem__(index=3)
# print(f'ehr_data: {ehr_data}')
# print(f'cxr_data: {cxr_data}')
# print(f'y: {y}')
# print(f'ech_data {ehr_data}')
# print(f'cxr_data {cxr_data}')
# print(f'y {y}')
# logger.info(f'EHR Data: {ehr_data}')
# logger.info(f'CXR Data: {cxr_data}')
# logger.info(f'Label (y): {y}')
# train_ds=MIMIC_CSR_EHR(ehr_ds,cxr_ds,split='train')

#------------------------------------------------------------
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
from scipy.signal import stft
import os
import sys
# 将项目根目录添加到 Python 路径
sys.path.append(os.path.abspath('/home/mimic/MIMIC_subset/MIMIC_subset'))
# from dataset.ECG_dataset import get_ECG_datasets,get_data_loader

# print(a)
from argument import args_parser
parser = args_parser()
# add more arguments here ...
args = parser.parse_args()
# from dataset.ECG_dataset import get_ECG_datasets,get_data_loader

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
    new_sig = np.pad(new_sig, ((48,48),(0,0)), mode='constant')
    new_sig = new_sig.astype(np.float32) # store as float32, else OOM
    return new_sig
    
def generate_file_path(base_path, split):
    # 使用 f-string 生成路径
    return f"{base_path}{split}.csv"

def generate_file_path(base_path, split):
    # 使用 f-string 生成路径
    return f"{base_path}{split}.csv"

def get_image_path(cxr_rootpath, dicom_id):
    # 构造完整的图像路径
    image_filename = f"{dicom_id}.jpg"
    image_path = os.path.join(cxr_rootpath, image_filename)
    return image_path

class MIMIC_ECG_CXR(Dataset):
    def __init__(self,  fullinfo_file,transform_i=None,transform_e=None, split='train'):#delete args in params
        # self.data_dir = args.cxr_data_dir
        # cxr_data_dir='/data/ke/data/MIMIC/physionet.org/files/mimic-cxr-jpg/2.0.0'
        # fullinfo_file='/data/ke/MIMIC_subset/PA_subset/with_nonan_label_PA_'
        self.args = args
    #     self.CLASSES  = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    #    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    #    'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
    #    'Pneumonia', 'Pneumothorax', 'Support Devices']\
        self.split=split
        # self.module = module
        self.fullinfo_path=fullinfo_file
        file_path = generate_file_path(fullinfo_file, self.split)
        # print(file_path)#/data/ke/MIMIC_subset/PA_subset/with_nonan_label_PA_train.csv
        # self.ecg_dir='/data/ke/data/physionet.org/files/mimic-iv-ecg/1.0'
        self.ecg_dir='/home/mimic/MIMIC_subset/MIMIC_subset/ECG_data'
        self.cxr_rootpath='/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC_CXRdataset'

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
                'dicom_id': mas[14].strip(),
                }
                for mas in self._data
        }
        print(len(self.data_map))#train 10632;test/val:1653
        self.names = list(self.data_map.keys())
        # print(self.data_map)
        self.transform_e = transform_e
        self.transform_i=transform_i

    def __getitem__(self, index):
        # print(f'get_item index {index}')
        if isinstance(index, int):
            index1 = self.names[index]
            # print(index)#is hadm_id
            y = self.data_map[index1]['labels']
            # print(y)
            study_whole_path = self.data_map[index1]['first_ecg_path']
            study_id_path='/'.join(study_whole_path.split('/')[-2:])
            rec_path=f'{self.ecg_dir}/{study_id_path}'
            # print(f'get item path {rec_path}')
            rd_record = wfdb.rdrecord(rec_path)
            

            signal = rd_record.p_signal # 5k x 12 signal  #not none


            if np.isnan(signal).any():
                return None  # 返回 None 或者其他标记值，表示该实例无效

            signal=adjust_sig(signal)#4096,12
            signal = torch.tensor(signal, dtype=torch.float32)
            signal=signal.permute(1,0)

            if self.args.domain=='frequency':
                # print('couculate frequency mean,std')
                # print(f'0 {signal.shape}') #[12,4096]
                f,t, Zxx = stft(signal,fs=100, window='hann',nperseg=25) #signal=[channel,dim]
                signal = np.abs(Zxx)
                # print(f'0.1 {signal.shape}')#[12,13,316]

                signal = signal.transpose(1,2,0)

                if self.split=='train':

                    mean=torch.tensor([0.012093465775251389, 0.01248111017048359, 0.011851412244141102, 0.010647540912032127, 0.01052943430840969, 0.010320836678147316, 0.01285327784717083, 0.017218537628650665, 0.018132127821445465, 0.016559477895498276, 0.014736325480043888, 0.012929477728903294])
                    std=torch.tensor([0.04458073899149895, 0.043323367834091187, 0.046114955097436905, 0.03714632987976074, 0.03872832655906677, 0.03995785862207413, 0.05534983053803444, 0.07524872571229935, 0.07799569517374039, 0.06808917224407196, 0.060315001755952835, 0.05352987349033356])

                    signal = torch.tensor(signal)
                    signal=(signal-mean) / std
                    signal = signal.permute(2, 0, 1)

                if self.split=='val':

                    signal = torch.tensor(signal)
                    mean=torch.tensor([0.012141992338001728, 0.01243192795664072, 0.011883892118930817, 0.010658721439540386, 0.010514732450246811, 0.010389219038188457, 0.012925531715154648, 0.01700308732688427, 0.018429512158036232, 0.016806283965706825, 0.014956677332520485, 0.012815584428608418])
                    std=torch.tensor([0.04507102072238922, 0.04274924471974373, 0.04663681611418724, 0.03698218986392021, 0.038571715354919434, 0.040647998452186584, 0.05551350489258766, 0.07260414958000183, 0.0793217197060585, 0.06911218911409378, 0.0626128762960434, 0.05248742550611496])
                    # signal=(signal-mean.view(-1,1)) / std.view(-1,1)
                    signal=(signal-mean) / std
                    signal = signal.permute(2, 0, 1)
            else:
#normalize
                if self.split=='train':

                    mean=torch.tensor([0.022676835, 0.013688515, -0.007821267, -0.017821848, 0.0023585667, 0.014657727, -0.018340515, -0.0038577598, -0.00035443218, 0.014264569, 0.020447042, 0.019408429])
                    std=torch.tensor([0.15746197, 0.15279943, 0.16141628, 0.13161984, 0.1358389, 0.1403546, 0.19144094, 0.2620128, 0.27269644, 0.24160655, 0.21423854, 0.18789752])

                    signal=(signal-mean.view(-1,1)) / std.view(-1,1)

                if self.split=='val':
                    mean=torch.tensor([0.022080505, 0.012505924, -0.008371157, -0.016918872, 0.0015192, 0.014522009, -0.018306602, -0.003076632, 0.000574659, 0.0146055035, 0.01795028, 0.02059472])
                    std=torch.tensor([0.16260016, 0.1545154, 0.16712894, 0.13399345, 0.13874556, 0.1459843, 0.19639233, 0.25812393, 0.2829126, 0.24955757, 0.22786401, 0.18867913])
                    signal=(signal-mean.view(-1,1)) / std.view(-1,1)

            dicom_id = self.data_map[index1]['dicom_id']
            img_path= get_image_path(self.cxr_rootpath, dicom_id)

            if not os.path.exists(img_path):
                # print(f'缺失文件 ID: {index1}, 路径: {img_path}')  # 打印缺失文件的 ID 和路径
                return None  # 可以返回 None 或者其他默认值


            img = Image.open(img_path).convert('RGB')






            if self.transform_e is not None:
                # signal = torch.tensor(signal, dtype=torch.float32)
                signal = self.transform_e(signal)
                
            if self.transform_i is not None:
                # signal = torch.tensor(signal, dtype=torch.float32)
                img = self.transform_i(img)
                

            # if self.module=='fusion':

            #     return {index:(signal, y)}
            # else:
                # print(f'model!=fusion signal {signal}')
                return signal,img,y



    def __len__(self):
        return len(self.data_map)
    

def get_transforms():#delete rgs param
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transforms = []
    train_transforms.append(transforms.Resize(256))
    train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.append(transforms.RandomAffine(degrees=45, scale=(.85, 1.15), shear=0, translate=(0.15, 0.15)))
    train_transforms.append(transforms.CenterCrop(224))
    train_transforms.append(transforms.ToTensor())
    train_transforms.append(normalize)      


    test_transforms = []
    test_transforms.append(transforms.Resize(256))


    test_transforms.append(transforms.CenterCrop(224))

    test_transforms.append(transforms.ToTensor())
    test_transforms.append(normalize)


    return train_transforms, test_transforms


def load_cxr_ecg_ds():
    train_transforms, test_transforms = get_transforms()#delete args param

    cxr_rootpath='/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC_CXRdataset'
    fullinfo_file='/home/mimic/MIMIC_subset/MIMIC_subset/PA_subset/with_nonan_label_PA_'

    dataset_train = MIMIC_ECG_CXR(fullinfo_file,  split='train', transform_i=transforms.Compose(train_transforms))#delete rgs param
    dataset_validate = MIMIC_ECG_CXR(fullinfo_file,  split='val', transform_i=transforms.Compose(test_transforms))#delete rgs param
    dataset_test = MIMIC_ECG_CXR(fullinfo_file, split='test', transform_i=transforms.Compose(test_transforms))#delete rgs param

    return dataset_train, dataset_validate, dataset_test

def get_ecgcxr_data_loader(batch_size):
    train_ds, val_ds, test_ds = load_cxr_ecg_ds()
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


    signal = np.stack([item[0] for item in batch])
    img = np.stack([item[1] for item in batch])

# 
    
    # 将目标标签（y）转换为NumPy数组
    targets = np.array([item[2] for item in batch])
    
    # 返回输入数据和对应的目标标签
    return [signal,img,targets]


    # ecg_train_ds, ecg_val_ds, ecg_test_ds = get_ECG_datasets(module='fusion')

    # cxr_train_ds, cxr_val_ds, cxr_test_ds = get_cxr_datasets(module='fusion')

    # train_ds = MIMIC_ECG_CXR(ecg_train_ds,cxr_train_ds,split='train')
    # val_ds = MIMIC_ECG_CXR(ecg_val_ds,cxr_val_ds,split='val')
    # test_ds = MIMIC_ECG_CXR(ecg_test_ds,cxr_test_ds,split='test')

    # train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=16, drop_last=True)
    # val_dl = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=my_collate, pin_memory=True, num_workers=16, drop_last=False)
    # test_dl = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=my_collate, pin_memory=True, num_workers=16, drop_last=False)

    # return train_dl, val_dl, test_dl

# def my_collate(batch):
#     # 将样本的输入数据（x）堆叠成一个批次
#     ehr_data = np.stack([item[0] for item in batch])
    
#     # 将目标标签（y）转换为NumPy数组
#     cxr_data = np.array([item[1] for item in batch])
#     target=
#     # 返回输入数据和对应的目标标签
#     return [ehr_data,cxr_data,targets]

def my_collate(batch):
    batch = [item for item in batch if item is not None]

    ecg_data = [item[0] for item in batch]
    # pairs = [False if item[1] is None else True for item in batch]
    img = torch.stack([item[1] for item in batch])
    # x, seq_length = pad_zeros(x)
    targets = np.array([item[2] for item in batch])
    # targets_cxr = torch.stack([torch.zeros(14) if item[3] is None else item[3] for item in batch])
    return [ecg_data,img,targets]


train_dl, val_dl = get_ecgcxr_data_loader(batch_size=16)

# 访问train_dl中的数据
# for batch_idx, batch in enumerate(train_dl):
#     # batch 是从 collate_fn 处理后的结果
#     signal,cxr_data, label = batch  # 这里假设 my_collate 函数返回(signal, label)
    
#     print(f"Batch {batch_idx}:")
#     print(f"Signal shape: {len(signal)}")  # 打印信号的形状
#     print(f'signal {type(signal)}')
    
#     tensor_signal = torch.tensor(signal)
#     print(f"tensor Signal shape: {tensor_signal.shape}")

#     print(f'cxr shape: {cxr_data.shape}')
#     print(f"Labels: {label.shape}")  # 打印标签





