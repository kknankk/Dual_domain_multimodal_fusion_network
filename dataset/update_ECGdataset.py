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
# import h5py
import torch # to set manual seed
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import wfdb # wfdb?
from scipy.signal import stft
import os
import sys
import numpy as np
from scipy.ndimage import zoom
# # 将项目根目录添加到 Python 路径
# sys.path.append(os.path.abspath('/data/mimic/MIMIC_subset/MIMIC_subset'))
# # from dataset.ECG_dataset import get_ECG_datasets,get_data_loader

# # print(a)
# from argument import args_parser

# 将 argument.py 所在的目录添加到 sys.path
sys.path.append(os.path.abspath('/home/mimic/MIMIC_subset/MIMIC_subset'))

# 然后再尝试导入
from argument import args_parser

parser = args_parser()
# add more arguments here ...
args = parser.parse_args()

def adjust_sig(signal):

    new_sig = sig.resample(signal, 4000, axis=0)
    #TODO: change （1，1）to（48，48）
    new_sig = np.pad(new_sig, ((48,48),(0,0)), mode='constant')
    new_sig = new_sig.astype(np.float32) # store as float32, else OOM
    return new_sig
    
def generate_file_path(base_path, split):
    # 使用 f-string 生成路径
    return f"{base_path}{split}.csv"

class MIMICECG(Dataset):
    def __init__(self,  fullinfo_file,transform=None, split='train',module='fusion'):#delete args in params

        self.args = args

        self.split=split
        self.module = module
        self.fullinfo_path=fullinfo_file
        file_path = generate_file_path(fullinfo_file, self.split)

        # self.ecg_dir='/home/mimic/MIMIC_subset/MIMIC_subset/ECG_data'
        self.ecg_dir='/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimic-iv-ecg/1.0'


        with open(file_path, "r")as lfile:
            self._data = lfile.readlines()
        self._listfile_header = self._data[0]

        self.CLASSES=self._listfile_header.strip().split(',')[6:13]
        self._data = self._data[1:]

        self._data = [line.split(',') for line in self._data]


        self.data_map = {}

        # self.data_map = {
        #     mas[1]: {
        #         #  'labels': [1 if label == '1' else 0 for label in mas[9:16]],
        #         'labels': [1 if mas[i] == '1' else 0 for i in [6, 7, 8, 9,10, 11,12]],

        #         'hadm_id': float(mas[1]),
        #         'first_ecg_path': mas[4],
        #         # 'dicom_id': float(mas[14]),
        #         }
        #         for mas in self._data
        # }
        self.data_map = {
            mas[1]: {
                # 'labels': [1 if label == '1' else 0 for label in mas[10,11,12,13]],
                'labels': [1 if mas[i] == '1' else 0 for i in [10, 11,12,13]],

                'hadm_id': float(mas[2]),
                # 'hadm_id': float(mas[1]),
                'ecg_path': mas[14].strip(),
                }
                for mas in self._data
        }

        self.data_map = {hadm_id: data for hadm_id, data in self.data_map.items() if any(data['labels'])}

        self.names = list(self.data_map.keys())

        self.transform = transform

    def __getitem__(self, index):
        # print(f'get_item index {index}')
        if isinstance(index, int):
            index1 = self.names[index]
            # print(index)#is hadm_id
            y = self.data_map[index1]['labels']

            # print(y)
            study_whole_path = self.data_map[index1]['ecg_path']
            # study_id_path='/'.join(study_whole_path.split('/')[-2:])
            # rec_path=f'{self.ecg_dir}/{study_id_path}'
            rec_path=f'{self.ecg_dir}/{study_whole_path}'
            # print(f'get item path {rec_path}')
            rd_record = wfdb.rdrecord(rec_path)

            signal = rd_record.p_signal # 5k x 12 signal  #not none

            signal=adjust_sig(signal)#4096,12
            for i in range(signal.shape[1]):
                # tmp = pd.DataFrame(signal[:,i]).interpolate().values.ravel().tolist()
                signal[:,i]= np.clip(signal[:,i],a_max=3, a_min=-3) 
            
            if np.isnan(signal).any():
                # print("Data contains NaN values. Dropping the entire data.")
                # 清空数据（或其他处理逻辑）
                signal = None  


            if signal is not None:
                signal = torch.tensor(signal, dtype=torch.float32)
                signal=signal.permute(1,0)

                if self.transform is not None:
                    if args.domain=='ecg_fusion':
                        temp_ecg,freq_ecg=self.transform(signal, self.split)
                        return index1,temp_ecg,freq_ecg,y
                    else:
                        signal = self.transform(signal, self.split)

                        return index1, signal, y

    def __len__(self):
        return len(self.data_map)


def get_transforms(domain=args.domain):
    if args.domain == 'frequency':
        return frequency_transforms()
    elif args.domain=='ecg_fusion':
        return fusion_frequency_transforms()
    else:
        return time_transforms()


def frequency_transforms():
    def transform(signal, split):
        # Frequency domain transformation logic
        # print(f'----------start frequency transform---------')
        f, t, Zxx = stft(signal, fs=400, window='hann', nperseg=512, noverlap=256)
        signal = np.abs(Zxx)
        signal = signal.transpose(1,2,0)
        signal = torch.tensor(signal)
        # print(f'frequency ecg {signal}')
       # Normalize signal based on split
       # #--------先注释
        if split == 'train':
            mean = torch.tensor([0.0027175110299140215, 0.0025571915321052074, 0.0027259790804237127, 0.0022376852575689554, 0.002268535317853093, 0.0024231092538684607, 0.0031389782670885324, 0.00440770061686635, 0.004576255101710558, 0.0041439891792833805, 0.0036769809667021036, 0.00312391621991992])
            std = torch.tensor([0.007386750541627407, 0.007428528741002083, 0.008423478342592716, 0.006059473846107721, 0.00703539839014411, 0.00704612024128437, 0.01027699001133442, 0.014795736409723759, 0.015199046581983566, 0.012593473307788372, 0.010705554857850075, 0.009277996607124805])
        else:  # Validation or test split
            mean = torch.tensor([0.0026735656429082155, 0.0025286851450800896, 0.0026889126747846603, 0.0022020991891622543, 0.002242852235212922, 0.002383109414950013, 0.003116166451945901, 0.004410209599882364, 0.004606673028320074, 0.004137404728680849, 0.003664759686216712, 0.003151838667690754])
            std = torch.tensor([0.007413068320602179, 0.007148303557187319, 0.00822953786700964, 0.005940577480942011, 0.0067643956281244755, 0.006992437411099672, 0.010407953523099422, 0.014992096461355686, 0.015441838651895523, 0.012707527726888657, 0.010589796118438244, 0.009325820952653885])

        # Normalize the signal
        signal = (signal - mean) / std
        signal = signal.permute(2, 0, 1)
        # print(f'fre signal {signal}')
        #--------先注释
        return signal

    return transform


def time_transforms():
    def transform(signal, split):
        # PRINT(F'---------TEMPORAL-----------------')
        # Time domain transformation logic
        #----先注释-----------
        if split == 'train':
            mean = torch.tensor([0.020118936896324158, 0.010715494863688946, -0.008904249407351017, -0.01507102232426405, 0.0005865496350452304, 0.014280002564191818, -0.020457008853554726, -0.014558318071067333, -0.010938421823084354, 0.005140588618814945, 0.015684371814131737, 0.01892842911183834])
            std = torch.tensor( [0.14524735510349274, 0.14451056718826294, 0.16435310244560242, 0.11869681626558304, 0.13675516843795776, 0.1381322145462036, 0.19892378151416779, 0.2894032895565033, 0.2974630296230316, 0.24795465171337128, 0.2104521095752716, 0.1804078221321106])
        #     mean = torch.tensor([0.020117154344916344, 0.010721409693360329, -0.008898481726646423,
        #                          -0.015055567026138306, 0.00059915566816926, 0.014294213615357876, 
        #                         -0.02039855159819126, -0.014887010678648949, -0.011265503242611885, 
        #                         0.005067931488156319, 0.01571229100227356, 0.018941625952720642])

        #     std = torch.tensor( [0.14577583968639374, 0.14529186487197876, 0.1657516062259674,
        #                          0.11890283972024918, 0.1377488523721695, 0.13876873254776, 
        #                          0.2000715583562851, 0.29409414529800415, 0.3019532859325409,
        #                           0.24997882544994354, 0.2122967392206192, 0.18199026584625244])
        else:  # Validation or test split
            mean = torch.tensor([0.020131081342697144, 0.011969772167503834, -0.007804600056260824, -0.015651753172278404, 0.0017433963948860765, 0.013781293295323849, -0.020989634096622467, -0.01432200986891985, -0.011121232062578201, 0.005424626171588898, 0.0167344119399786, 0.019787846133112907])
            std = torch.tensor([0.1451718956232071, 0.13993124663829803, 0.16126003861427307, 0.1166679635643959, 0.1325085312128067, 0.13701407611370087, 0.20065732300281525, 0.2904973328113556, 0.30150458216667175, 0.24946753680706024, 0.2085689753293991, 0.18243145942687988])
        #     mean = torch.tensor([0.02010100893676281, 0.011963431723415852, -0.007788896095007658, 
        #                         -0.015638411045074463, 0.0017529507167637348, 0.013761517591774464, 
        #                         -0.021034805104136467, -0.014610975980758667, -0.01137454342097044,
        #                          0.005358664784580469, 0.016654817387461662, 0.01987520232796669])

        #     std = torch.tensor([0.14540570974349976, 0.14031027257442474, 0.16209889948368073, 
        #                         0.11695944517850876, 0.133339986205101, 0.13741260766983032, 
        #                         0.20248259603977203, 0.29613280296325684, 0.30761823058128357,
        #                          0.25111329555511475, 0.21117143332958221, 0.18379004299640656])

        # # Normalize the signal
        # signal = (signal - mean.view(-1, 1)) / std.view(-1, 1)
         #----先注释-----------
        return signal

    return transform

def fusion_frequency_transforms():
    def transform(signal, split):
        # 选择时间实例的特定部分
        # time_instance = time_instance[100:4900, :]  # (4800, 12)
        # print(f'fusion ecg {signal.shape}')#[12,4096]

        # 短时傅里叶变换 (STFT)
        f, t, Zxx = stft(signal, fs=500, window='hann', nperseg=125)
        spectrogram_instance = np.abs(Zxx)  # (12, 63, 66)
        # print(f'spectrogram_instance {spectrogram_instance.shape}')
        spectrogram_instance = spectrogram_instance.transpose(1, 2, 0)  # (63, 78, 12)
        # print(f'spectrogram_instance {spectrogram_instance.shape}')
        signal=signal.transpose(0,1)
        # print(f'fusion ecg {signal.shape}')#[4096,12]
         # Normalize the time instance
        # time_instance = (signal - mean_signal.view(-1, 1)) / std_signal.view(-1, 1)

        return signal, spectrogram_instance

    return transform





def get_ECG_datasets(module='unimodal'):
    train_transforms = get_transforms(domain=args.domain)  # Specify the domain
    fullinfo_file='/home/mimic/MIMIC_subset/MIMIC_subset/PA_subset/with_nonan_label_PA_'

    dataset_train = MIMICECG(fullinfo_file, transform=train_transforms, split='train', module=module)
    dataset_validate = MIMICECG(fullinfo_file, transform=train_transforms, split='val', module=module)
    dataset_test = MIMICECG(fullinfo_file, transform=train_transforms, split='test', module=module)
    return dataset_train, dataset_validate, dataset_test


def get_data_loader(batch_size):
    train_ds, val_ds, test_ds = get_ECG_datasets(args)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=my_collate, pin_memory=True, num_workers=16)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False, collate_fn=my_collate, pin_memory=True, num_workers=16)

    return train_dl, val_dl

def my_collate(batch):
    # 将样本的输入数据（x）堆叠成一个批次
    #TODO: initial:
    if args.domain=='ecg_fusion':
        batch = [item for item in batch if item is not None]

        index1=np.stack([item[0] for item in batch])
        temporal_ecg = np.stack([item[1] for item in batch])
        frequency_ecg = np.stack([item[2] for item in batch])


        targets = np.array([item[3] for item in batch])
        return [index1,temporal_ecg,frequency_ecg, targets]
    else:
        batch = [item for item in batch if item is not None]

        index1=np.stack([item[0] for item in batch])
        x = np.stack([item[1] for item in batch])


        targets = np.array([item[2] for item in batch])
    
    # 返回输入数据和对应的目标标签
        return [index1,x, targets]

# ehr_train,ehr_test,ehr_val=get_ECG_datasets(args)

# train_ecg_dl, val_ecg_dl = get_data_loader(batch_size=args.batch_size)

# for i,data in enumerate(ehr_train):
#     print(i,data)
