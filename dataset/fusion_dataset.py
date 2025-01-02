



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
   
    return f"{base_path}{split}.csv"

def generate_file_path(base_path, split):

    return f"{base_path}{split}.csv"

def get_image_path(cxr_rootpath, dicom_id):

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
        # self.ecg_dir='/home/mimic/MIMIC_subset/MIMIC_subset/ECG_data'
        # self.cxr_rootpath='/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC_CXRdataset'
        self.cxr_rootpath='/home/mimic/MIMIC_subset/MIMIC_subset/resized'
        self.ecg_dir='/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimic-iv-ecg/1.0'

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

        # self.data_map = {
        #     mas[1]: {
        #         'labels': [1 if label == '1.0' else 0 for label in mas[6:13]],
        #         'hadm_id': float(mas[1]),
        #         'first_ecg_path': mas[4],
        #         'dicom_id': mas[14].strip(),
        #         }
        #         for mas in self._data
        # }
        self.data_map = {
            mas[1]: {
                # 'labels': [1 if label == '1' else 0 for label in mas[10,11,12,13]],
                'labels': [1 if mas[i] == '1' else 0 for i in [10, 11,12,13]],

                'hadm_id': float(mas[2]),
                # 'hadm_id': float(mas[1]),
                'dicom_id': mas[16].strip(),
                'ecg_path': mas[14].strip(),
                }
                for mas in self._data
        }
        # print(len(self.data_map))#train 10632;test/val:1653
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
            study_whole_path = self.data_map[index1]['ecg_path']
            # study_id_path='/'.join(study_whole_path.split('/')[-2:])
            dicom_id = self.data_map[index1]['dicom_id']

            img_path= get_image_path(self.cxr_rootpath, dicom_id)
            # print(f'get cxr path {img_path}')
            rec_path=f'{self.ecg_dir}/{study_whole_path}'
            # print(f'get ecg path {rec_path}')
            rd_record = wfdb.rdrecord(rec_path)
            

            signal = rd_record.p_signal # 5k x 12 signal  #not none


            if np.isnan(signal).any():
                return None  

            signal=adjust_sig(signal)#4096,12
            for i in range(signal.shape[1]):
                # tmp = pd.DataFrame(signal[:,i]).interpolate().values.ravel().tolist()
                signal[:,i]= np.clip(signal[:,i],a_max=3, a_min=-3) 
            
            if np.isnan(signal).any():
                print("Data contains NaN values. Dropping the entire data.")
               
                signal = None  

            if signal is not None:

                signal = torch.tensor(signal, dtype=torch.float32)
                signal=signal.permute(1,0)

            if self.args.domain=='frequency' :
                # print('couculate frequency mean,std')
                # print(f'0 {signal.shape}') #[12,4096]
                # f,t, Zxx = stft(signal,fs=100, window='hann',nperseg=512,noverlap=256) #signal=[channel,dim]
                if self.args.fusion_type=='deeper_frequency_fusion':
                    # f,t, Zxx = stft(signal,fs=100, window='hann',nperseg=512,noverlap=256) #signal=[channel,dim]

                    # signal=Zxx
                    signal=signal
                    # print(f'without abs signal {signal.shape}')#[12,257,17]
                    # is_complex = np.iscomplexobj(signal)
                    # print(f"Is the signal data complex? {is_complex}") #True

                else:
                    f,t, Zxx = stft(signal,fs=100, window='hann',nperseg=512,noverlap=256) #signal=[channel,dim]

                    signal = np.abs(Zxx) #[B,channel,257,17]
                
                if self.args.domain=='frequency' and self.args.fusion_type=='deeper_frequency_fusion' :
                    signal=signal.transpose(1,0)
                else:    
                    signal = signal.transpose(1,2,0)
                # print(f'0.1 {signal.shape}')#[257,17,12]

                if self.split=='train' and  self.args.domain!='deeper_frequency_fusion':
                    # print(f'frequency train norm')#Y

                    mean=torch.tensor([0.012093465775251389, 0.01248111017048359, 0.011851412244141102, 0.010647540912032127, 0.01052943430840969, 0.010320836678147316, 0.01285327784717083, 0.017218537628650665, 0.018132127821445465, 0.016559477895498276, 0.014736325480043888, 0.012929477728903294])
                    std=torch.tensor([0.04458073899149895, 0.043323367834091187, 0.046114955097436905, 0.03714632987976074, 0.03872832655906677, 0.03995785862207413, 0.05534983053803444, 0.07524872571229935, 0.07799569517374039, 0.06808917224407196, 0.060315001755952835, 0.05352987349033356])

                    signal = torch.tensor(signal)
    
                    signal=(signal-mean) / std
                    # print(f'fusion_type {self.args.fusion_type}')
                    if self.args.fusion_type=='deeper_frequency_fusion':
                        signal=signal
                        # print(f'after norm ecg {signal.shape}')#【257，17，12】
                    else:

                        signal = signal.permute(2, 0, 1)

                if self.split=='val' and self.args.domain!='deeper_frequency_fusion':

                    signal = torch.tensor(signal)
                    # signal = signal.clone().detach()
                    mean=torch.tensor([0.012141992338001728, 0.01243192795664072, 0.011883892118930817, 0.010658721439540386, 0.010514732450246811, 0.010389219038188457, 0.012925531715154648, 0.01700308732688427, 0.018429512158036232, 0.016806283965706825, 0.014956677332520485, 0.012815584428608418])
                    std=torch.tensor([0.04507102072238922, 0.04274924471974373, 0.04663681611418724, 0.03698218986392021, 0.038571715354919434, 0.040647998452186584, 0.05551350489258766, 0.07260414958000183, 0.0793217197060585, 0.06911218911409378, 0.0626128762960434, 0.05248742550611496])
                    # signal=(signal-mean.view(-1,1)) / std.view(-1,1)
                    signal=(signal-mean) / std
                    # signal = signal.permute(2, 0, 1)
                    if self.args.fusion_type=='deeper_frequency_fusion':
                        signal=signal
                    else:

                        signal = signal.permute(2, 0, 1)

                if self.split=='train' and  self.args.domain=='deeper_frequency_fusion':
                    # print(f'frequency train norm')#Y

                    # mean=torch.tensor([0.022676835, 0.013688515, -0.007821267, -0.017821848, 0.0023585667, 0.014657727, -0.018340515, -0.0038577598, -0.00035443218, 0.014264569, 0.020447042, 0.019408429])
                    # std=torch.tensor([0.15746197, 0.15279943, 0.16141628, 0.13161984, 0.1358389, 0.1403546, 0.19144094, 0.2620128, 0.27269644, 0.24160655, 0.21423854, 0.18789752])
                    
                    # mean = torch.tensor([0.020118936896324158, 0.010715494863688946, -0.008904249407351017, -0.01507102232426405, 0.0005865496350452304, 0.014280002564191818, -0.020457008853554726, -0.014558318071067333, -0.010938421823084354, 0.005140588618814945, 0.015684371814131737, 0.01892842911183834])
                    # std = torch.tensor( [0.14524735510349274, 0.14451056718826294, 0.16435310244560242, 0.11869681626558304, 0.13675516843795776, 0.1381322145462036, 0.19892378151416779, 0.2894032895565033, 0.2974630296230316, 0.24795465171337128, 0.2104521095752716, 0.1804078221321106])

                    # signal = torch.tensor(signal)
                    # signal=(signal-mean) / std
                    # print(f'fusion_type {self.args.fusion_type}')
                    signal=signal
                    if self.args.fusion_type=='deeper_frequency_fusion':
                        signal=signal
                        # print(f'after norm ecg {signal.shape}')#【257，17，12】
                    else:

                        signal = signal.permute(2, 0, 1)

                if self.split=='val' and self.args.domain=='deeper_frequency_fusion':

                    # signal = torch.tensor(signal)
                   
                    # signal = signal.clone().detach()
                    # mean = torch.tensor([0.020131081342697144, 0.011969772167503834, -0.007804600056260824, -0.015651753172278404, 0.0017433963948860765, 0.013781293295323849, -0.020989634096622467, -0.01432200986891985, -0.011121232062578201, 0.005424626171588898, 0.0167344119399786, 0.019787846133112907])
                    # std = torch.tensor([0.1451718956232071, 0.13993124663829803, 0.16126003861427307, 0.1166679635643959, 0.1325085312128067, 0.13701407611370087, 0.20065732300281525, 0.2904973328113556, 0.30150458216667175, 0.24946753680706024, 0.2085689753293991, 0.18243145942687988])
                    # # signal=(signal-mean.view(-1,1)) / std.view(-1,1)
                    # signal=(signal-mean) / std
                 
                    # signal = signal.permute(2, 0, 1)
                    signal=signal
                    if self.args.fusion_type=='deeper_frequency_fusion':
                        signal=signal
                    else:

                        signal = signal.permute(2, 0, 1)

                    
            else:
#normalize
                if self.split=='train':
                    # print(f'S_T train norm---------------')

                    # mean=torch.tensor([0.021983036771416664, 0.013289721682667732, -0.007492052856832743, -0.0172954760491848, 0.002315341029316187, 0.014129746705293655, -0.018112635239958763, -0.0034318040125072002, 0.00021024956367909908, 0.014880170114338398, 0.02024325355887413, 0.01977406069636345])
                    # std=torch.tensor([0.15690241754055023, 0.15237219631671906, 0.15942928194999695, 0.13163667917251587, 0.13473273813724518, 0.13911446928977966, 0.18939709663391113, 0.25870004296302795, 0.2708594501018524, 0.2404349446296692, 0.21415464580059052, 0.18362797796726227])
                    mean = torch.tensor([0.020118936896324158, 0.010715494863688946, -0.008904249407351017, -0.01507102232426405, 0.0005865496350452304, 0.014280002564191818, -0.020457008853554726, -0.014558318071067333, -0.010938421823084354, 0.005140588618814945, 0.015684371814131737, 0.01892842911183834])
                    std = torch.tensor( [0.14524735510349274, 0.14451056718826294, 0.16435310244560242, 0.11869681626558304, 0.13675516843795776, 0.1381322145462036, 0.19892378151416779, 0.2894032895565033, 0.2974630296230316, 0.24795465171337128, 0.2104521095752716, 0.1804078221321106])

                    mean=mean.view(-1,1)

                    signal=(signal-mean.view(-1,1)) / std.view(-1,1)

                if self.split=='val':

                    # mean=torch.tensor([0.02482433430850506, 0.015947148203849792, -0.007936287671327591, -0.019916854798793793, 0.0034988929983228445, 0.01582620106637478, -0.019720222800970078, -0.004102816805243492, -0.0021764931734651327, 0.012111539952456951, 0.021004941314458847, 0.02135436423122883])
                    # std=torch.tensor([0.1607745736837387, 0.15428286790847778, 0.16370441019535065, 0.1342308074235916, 0.13679346442222595, 0.14322689175605774, 0.1934480518102646, 0.2608761787414551, 0.27516594529151917, 0.24000638723373413, 0.2119644731283188, 0.18333952128887177])
                    mean = torch.tensor([0.020131081342697144, 0.011969772167503834, -0.007804600056260824, -0.015651753172278404, 0.0017433963948860765, 0.013781293295323849, -0.020989634096622467, -0.01432200986891985, -0.011121232062578201, 0.005424626171588898, 0.0167344119399786, 0.019787846133112907])
                    std = torch.tensor([0.1451718956232071, 0.13993124663829803, 0.16126003861427307, 0.1166679635643959, 0.1325085312128067, 0.13701407611370087, 0.20065732300281525, 0.2904973328113556, 0.30150458216667175, 0.24946753680706024, 0.2085689753293991, 0.18243145942687988])

                    signal=(signal-mean.view(-1,1)) / std.view(-1,1)

            # dicom_id = self.data_map[index1]['dicom_id']
            # img_path= get_image_path(self.cxr_rootpath, dicom_id)

            if not os.path.exists(img_path):
               
                return None  


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
                # print(f'before return signal {signal.shape}')
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
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=32)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False, collate_fn=my_collate, pin_memory=True, num_workers=32)

    return train_dl, val_dl

# def my_collate(batch):
  
#     #TODO: initial:
#     batch = [item for item in batch if item is not None]

#     # for item in batch:
#     #     print(item[0].shape)
#     #     batch.append(item) 


#     signal = np.stack([item[0] for item in batch])
#     img = np.stack([item[1] for item in batch])

# # 
    

#     targets = np.array([item[2] for item in batch])
    

#     return [signal,img,targets]


    # ecg_train_ds, ecg_val_ds, ecg_test_ds = get_ECG_datasets(module='fusion')

    # cxr_train_ds, cxr_val_ds, cxr_test_ds = get_cxr_datasets(module='fusion')

    # train_ds = MIMIC_ECG_CXR(ecg_train_ds,cxr_train_ds,split='train')
    # val_ds = MIMIC_ECG_CXR(ecg_val_ds,cxr_val_ds,split='val')
    # test_ds = MIMIC_ECG_CXR(ecg_test_ds,cxr_test_ds,split='test')

    # train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=16, drop_last=True)
    # val_dl = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=my_collate, pin_memory=True, num_workers=16, drop_last=False)
    # test_dl = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=my_collate, pin_memory=True, num_workers=16, drop_last=False)

    # return train_dl, val_dl, test_dl


def my_collate(batch):
    batch = [item for item in batch if item is not None]

    ecg_data = [item[0] for item in batch]
    # pairs = [False if item[1] is None else True for item in batch]
    pairs = [False if item[1] is None else True for item in batch]

    img = torch.stack([item[1] for item in batch])
    # x, seq_length = pad_zeros(x)

    seq_length = [x.shape[0] for x in ecg_data]

    # max_len = max(seq_length)
    targets = np.array([item[2] for item in batch])
    # targets_cxr = torch.stack([torch.zeros(14) if item[3] is None else item[3] for item in batch])
    return [ecg_data,img,targets,seq_length,pairs]

# def my_collate_cxr_ehr(batch):
#     #x: all ehr data
#     x = [item[0] for item in batch]
#     #pairs: False if no cxr. True if cxr exists
#     pairs = [False if item[1] is None else True for item in batch]
#     # if cxr missing: use 0.
#     img = torch.stack([torch.zeros(3, 224, 224) if item[1] is None else item[1] for item in batch])
#     x, seq_length = pad_zeros(x)
#     targets_ehr = np.array([item[2] for item in batch]) #ehr labels
#     # targets_cxr = torch.stack([torch.zeros(14) if item[3] is None else item[3] for item in batch]) #cxr labels
#     return [x, img, targets_ehr, seq_length, pairs]


# train_dl, val_dl = get_ecgcxr_data_loader(batch_size=16)

# for i ,data in enumerate(train_dl):
#     print(i,data)






