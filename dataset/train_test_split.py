# import pandas as pd
# from sklearn.model_selection import train_test_split
# import os
# from skmultilearn.model_selection import iterative_train_test_split

# # X_train, y_train, X_test, y_test = iterative_train_test_split(x, y, test_size = 0.1)
# #/data/ke/MIMIC_subset/PA_subset
# # 文件路径
# input_csv_path = '/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/test1.csv'
# # input_csv_path ='/home/mimic/MIMIC_subset/MIMIC_subset/with_notransfer_nonan_label_PA.csv'
# # input_csv_path ='/home/mimic/MIMIC_subset/MIMIC_subset/ICU_WALKIN.csv'
# train_folder = '/home/mimic/MIMIC_subset/MIMIC_subset/PA_subset'
# val_folder = '/home/mimic/MIMIC_subset/MIMIC_subset/PA_subset'
# test_folder = '/home/mimic/MIMIC_subset/MIMIC_subset/PA_subset'

# # 加载数据
# data = pd.read_csv(input_csv_path)
# diseases = [
#     # 'Chronic obstructive pulmonary disease and bronchiectasis',
#     # 'Congestive heart failure; nonhypertensive',
#     # 'Coronary atherosclerosis and other heart disease',
#     # 'Essential hypertension',
#     # 'Hypertension with complications and secondary hypertension',
#     # 'Other lower respiratory disease',
#     # 'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)'
#     'Congestive heart failure; nonhypertensive',
#     'Coronary atherosclerosis and other heart disease',
#     'Hypertension with complications and secondary hypertension',
#     'Pneumonia_or_COPD'
# ]
# data[diseases] = data[diseases].fillna(0)
# y = data[diseases] 
# # print(data[diseases].sum())
# print(data.shape)
# print(y.shape)
# # 划分数据集为训练集（80%）和临时集（20%）
# # train_data, temp_data = train_test_split(data, test_size=0.2, random_state=0)
# train_data,train_label, temp_data,temp_label = iterative_train_test_split(data.values,y.values ,test_size=0.2)
# # 划分临时集为验证集（50%）和测试集（50%）
# # val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=0)
# val_data,val_label, test_data,test_label = iterative_train_test_split(temp_data,temp_label, test_size=0.5)

# # 创建文件夹，如果不存在的话
# os.makedirs(train_folder, exist_ok=True)
# os.makedirs(val_folder, exist_ok=True)
# os.makedirs(test_folder, exist_ok=True)

# train_data = pd.DataFrame(train_data, columns=data.columns)
# val_data = pd.DataFrame(val_data, columns=data.columns)
# test_data = pd.DataFrame(test_data, columns=data.columns)


# # 保存数据到相应的文件夹
# train_data.to_csv(os.path.join(train_folder, 'with_nonan_label_PA_train.csv'), index=False)
# val_data.to_csv(os.path.join(val_folder, 'with_nonan_label_PA_val.csv'), index=False)
# test_data.to_csv(os.path.join(test_folder, 'with_nonan_label_PA_test.csv'), index=False)

# print("Data has been split and saved successfully.")

import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 文件路径
input_csv_path = '/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/new3.csv'
train_folder = '/home/mimic/MIMIC_subset/MIMIC_subset/PA_subset'
val_folder = '/home/mimic/MIMIC_subset/MIMIC_subset/PA_subset'
test_folder = '/home/mimic/MIMIC_subset/MIMIC_subset/PA_subset'

# 加载数据
data = pd.read_csv(input_csv_path)
diseases = [
    # 'Congestive heart failure; nonhypertensive',
    # 'Coronary atherosclerosis and other heart disease',
    # 'Hypertension with complications and secondary hypertension',
    # 'Pneumonia_or_COPD'
    'Congestive heart failure; nonhypertensive',
    'Coronary atherosclerosis and other heart disease',
    'Pneumonia_or_COPD_or_OLRD_167',
    'EH_OR_HWCASH_45'
]
data[diseases] = data[diseases].fillna(0)
y = data[diseases]

# 创建一个标签列，确保每个标签的组合都保持比例
combined_labels = y.apply(lambda x: ''.join(map(str, x)), axis=1)

# 划分数据集
train_data, temp_data = train_test_split(data, test_size=0.2, stratify=combined_labels, random_state=0)
val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data[diseases].apply(lambda x: ''.join(map(str, x)), axis=1), random_state=0)

# 创建文件夹
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# 保存数据到相应的文件夹
train_data.to_csv(os.path.join(train_folder, 'with_nonan_label_PA_train.csv'), index=False)
val_data.to_csv(os.path.join(val_folder, 'with_nonan_label_PA_val.csv'), index=False)
test_data.to_csv(os.path.join(test_folder, 'with_nonan_label_PA_test.csv'), index=False)

print("Data has been split and saved successfully.")
