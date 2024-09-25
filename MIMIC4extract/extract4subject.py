# import pandas as pd
#-!-!-!-!-!_!-!-!-!__!-!-!-!-!-!-_!-！move mimicdataset from /data/ke/MedFuse/mimic4extract/data/physionet.org/files.....to /data/ke/data/physionet.org/files
#TODO:change data_path
from mimic3csv import *
# # 定义文件路径
# cxr_metadata_path = '/data/ke/MedFuse/mimic4extract/data/MIMIC/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv'
# ecg_record_list_path = '/data/ke/MedFuse/mimic4extract/data/physionet.org/files/mimic-iv-ecg/1.0/record_list.csv'
# output_path = '/data/ke/MIMIV_subset/same_subjectid.csv'

# # 读取数据
# cxr_metadata = pd.read_csv(cxr_metadata_path, usecols=['subject_id'])
# ecg_record_list = pd.read_csv(ecg_record_list_path, usecols=['subject_id'])

# # 找到相同的subject_id
# same_subject_ids = pd.merge(cxr_metadata, ecg_record_list, on='subject_id')

# # 删除重复项（如果需要）
# same_subject_ids = same_subject_ids.drop_duplicates()

# # 保存到新的CSV文件
# same_subject_ids.to_csv(output_path, index=False)

# print(f'Saved {len(same_subject_ids)} common subject_id(s) to {output_path}')


#above are code to extract '/data/ke/MIMIV_subset/same_subjectid.csv'
#-----------------------------------------------

#------------extract hamd_id, admittime, dischtime from admission.csv
# import pandas as pd

# # 读取 same_subjectid.csv 文件
# same_subjectid_path = '/data/ke/MIMIC_subset/same_subjectid.csv'
# same_subjectid = pd.read_csv(same_subjectid_path)

# # 读取 addmision.csv 文件
# addmission_path = '/data/ke/MedFuse/mimic4extract/data/MIMIC/physionet.org/files/mimiciv/1.0/core/admissions.csv'
# addmision = pd.read_csv(addmission_path)

# # 合并两个数据集，根据 subject_id 查找对应的 stay_id
# merged_df = same_subjectid.merge(addmision[['subject_id', 'hadm_id','admittime','dischtime']], how='left', on='subject_id')

# # 保存结果到新的 CSV 文件
# output_path = '/data/ke/MIMIC_subset/same_subjectid_with_hadm_id.csv'
# merged_df.to_csv(output_path, index=False)

# print(f"Updated file with stay_id saved to: {output_path}")
#----------------------------------------------------

#-----------extract first ECG with each hamd_id from record_list
# import pandas as pd

# # 读取数据
# same_subjectid_path = '/data/ke/MIMIC_subset/same_subjectid_with_hadm_id.csv'
# record_list_path = '/data/ke/MedFuse/mimic4extract/data/physionet.org/files/mimic-iv-ecg/1.0/record_list.csv'

# same_subjectid = pd.read_csv(same_subjectid_path)
# record_list = pd.read_csv(record_list_path)

# # 转换时间格式
# same_subjectid['admittime'] = pd.to_datetime(same_subjectid['admittime']).dt.date
# same_subjectid['dischtime'] = pd.to_datetime(same_subjectid['dischtime']).dt.date
# record_list['ecg_time'] = pd.to_datetime(record_list['ecg_time']).dt.date

# # 创建一个空的列表用于存储结果
# paths = []

# # 遍历每一行的 subject_id 和 admittime, dischtime
# for _, row in same_subjectid.iterrows():
#     subject_id = row['subject_id']
#     admittime = row['admittime']
#     dischtime = row['dischtime']
    
#     # 筛选出对应 subject_id 的 ECG 记录
#     subject_ecgs = record_list[record_list['subject_id'] == subject_id]
    
#     # 筛选出在 admittime 和 dischtime 之间的 ECG 记录
#     valid_ecgs = subject_ecgs[(subject_ecgs['ecg_time'] >= admittime) & (subject_ecgs['ecg_time'] <= dischtime)]
    
#     # 找到第一个符合条件的 ECG 记录的 path
#     if not valid_ecgs.empty:
#         first_ecg_path = valid_ecgs.iloc[0]['path']
#     else:
#         first_ecg_path = ''
    
#     paths.append(first_ecg_path)

# # 将结果添加到 same_subjectid 数据框中
# same_subjectid['first_ecg_path'] = paths

# # 保存到新的 CSV 文件
# output_path = '/data/ke/MIMIC_subset/hadmid_with_ECG.csv'
# same_subjectid.to_csv(output_path, index=False)

# print(f"Updated file with ECG path saved to: {output_path}")

#--------------------------------------------------
#-------------extract first CXR study_id from cxr-metadata.csv on samne subject_id & hadm_id

# import pandas as pd

# # 读取数据
# same_subjectid_path = '/data/ke/MIMIC_subset/hadmid_with_ECG.csv'
# record_list_path = '/data/ke/MedFuse/mimic4extract/data/MIMIC/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv'

# same_subjectid = pd.read_csv(same_subjectid_path)
# record_list = pd.read_csv(record_list_path)

# # 转换时间格式
# same_subjectid['admittime'] = pd.to_datetime(same_subjectid['admittime']).dt.date
# same_subjectid['dischtime'] = pd.to_datetime(same_subjectid['dischtime']).dt.date
# record_list['StudyDate'] = pd.to_datetime(record_list['StudyDate'],format='%Y%m%d').dt.date

# # 创建一个空的列表用于存储结果
# print(record_list['StudyDate'])
# paths = []

# # 遍历每一行的 subject_id 和 admittime, dischtime
# for _, row in same_subjectid.iterrows():
#     subject_id = row['subject_id']
#     admittime = row['admittime']
#     dischtime = row['dischtime']
    
#     # 筛选出对应 subject_id 的 ECG 记录
#     subject_ecgs = record_list[record_list['subject_id'] == subject_id]
    
#     # 筛选出在 admittime 和 dischtime 之间的 ECG 记录
#     valid_ecgs = subject_ecgs[(subject_ecgs['StudyDate'] >= admittime) & (subject_ecgs['StudyDate'] <= dischtime)]
    
#     # 找到第一个符合条件的 ECG 记录的 path
#     if not valid_ecgs.empty:
#         first_CXR_studyid = valid_ecgs.iloc[0]['study_id']
#     else:
#         first_CXR_studyid = ''
    
#     paths.append(first_CXR_studyid)

# # 将结果添加到 same_subjectid 数据框中
# same_subjectid['first_CXR_studyid'] = paths

# # 保存到新的 CSV 文件
# output_path = '/data/ke/MIMIC_subset/initial_hadmid_with_ECG_and_CXR.csv'
# same_subjectid.to_csv(output_path, index=False)

# print(f"Updated file with ECG path saved to: {output_path}")
# # 
#-------------------------------------------------------

#--------------extract both ecg+cxr from hadm_id_with_ECG_and_CXR
# import pandas as pd

# # 读取数据
# input_path = '/data/ke/MIMIC_subset/hadmid_with_ECG_and_CXR.csv'
# output_path = '/data/ke/MIMIC_subset/hadmid_with_ECG_and_CXR_nonan.csv'

# # 读取 CSV 文件
# data = pd.read_csv(input_path)

# # 筛选出 'first_ecg_path' 和 'first_CXR_studyid' 列都不为空的行
# filtered_data = data.dropna(subset=['first_ecg_path', 'first_CXR_studyid'])

# # 保存到新的 CSV 文件
# filtered_data.to_csv(output_path, index=False)

# print(f"Filtered data saved to: {output_path}")
#------------------------------------------------------------
# filter with the hadm_id who has icd_code

# import pandas as pd

# # Define file paths
# file_a_path = '/data/ke/MIMIC_subset/hadmid_with_ECG_and_CXR_nonan.csv'
# file_b_path = '/data/ke/MedFuse/mimic4extract/data/MIMIC/physionet.org/files/mimiciv/1.0/hosp/diagnoses_icd.csv'
# output_path = '/data/ke/MIMIC_subset/hadmid_with_ECG_and_CXR_nonan_hasicd.csv'

# # Read the CSV files
# df_a = pd.read_csv(file_a_path, dtype={'hadm_id': str})
# df_b = pd.read_csv(file_b_path, dtype={'hadm_id': str})

# # Normalize hadm_id by removing the decimal point and anything after it
# df_a['hadm_id'] = df_a['hadm_id'].astype(str).str.split('.').str[0]

# # Find hadm_id in both datasets
# common_hadm_ids = df_a[df_a['hadm_id'].isin(df_b['hadm_id'])]

# # Save the filtered rows to a new CSV file
# common_hadm_ids.to_csv(output_path, index=False)

# print(f"Rows with common hadm_id saved to: {output_path}")

#------------------------------
#delete the .(decimal) after hadm_id 
# import pandas as pd

# # 定义文件路径
# input_file_path = '/data/ke/MIMIC_subset/hadmid_with_ECG_and_CXR_nonan_hasicd.csv'
# output_file_path = './hadmid_with_ECG_and_CXR_nonan_hasicd_no_decimal.csv'

# # 读取CSV文件
# df = pd.read_csv(input_file_path, dtype={'hadm_id': str})

# # 规范化 hadm_id，去掉小数点后面的部分，只保留小数点前面的部分
# df['hadm_id'] = df['hadm_id'].astype(str).str.split('.').str[0]

# # 保存修改后的结果到新的CSV文件
# df.to_csv(output_file_path, index=False)

# print(f"File with hadm_id modified saved to: {output_file_path}")


#-------------------------------------------------------
# get icd code for each hadm_id save to all_diagnoses.csv
# import pandas as pd

# # 定义文件路径
# hadm_id_with_EC_hasicd_path = '/data/ke/MIMIC_subset/hadmid_with_ECG_and_CXR_nonan_hasicd_no_decimal.csv'
# diagnosis_path = '/data/ke/MedFuse/mimic4extract/data/MIMIC/physionet.org/files/mimiciv/1.0/hosp/diagnoses_icd.csv'
# output_path = '/data/ke/MIMIC_subset/root/all_diagnoses.csv'

# # 读取数据
# hadm_id_with_CE = pd.read_csv(hadm_id_with_EC_hasicd_path)
# diagnoses = pd.read_csv(diagnosis_path)

# # 合并 DataFrames 以提取 hadm_id 在 hadm_id_with_CE 中的行
# filtered_diagnoses = diagnoses.merge(hadm_id_with_CE[['subject_id', 'hadm_id']], how='inner',
#                                       left_on=['subject_id', 'hadm_id'], right_on=['subject_id', 'hadm_id'])

# # 保留重复的 hadm_id，并只保留 diagnoses 中的列
# filtered_diagnoses = filtered_diagnoses[diagnoses.columns]

# # 将结果保存到 CSV 文件
# filtered_diagnoses.to_csv(output_path, index=False)
#--------------------------------------------------------
#each subject has its folder with subject_id_diagnosis.csv
# import pandas as pd
# import yaml

# all_diagnoses_path='/data/ke/MIMIC_subset/root/all_diagnoses.csv'
# diagnoses=pd.read_csv(all_diagnoses_path)
# phenotype_definitions_path='/data/ke/MIMIC_subset/category2icd_code9_10.yaml'
# output_path='/data/ke/MIMIC_subset/root'
# phenotypes = add_hcup_ccs_2015_groups(diagnoses, yaml.load(open(phenotype_definitions_path, 'r'),Loader=yaml.FullLoader))
# stays_path='/data/ke/MIMIC_subset/root/all_diagnoses.csv'
# stays=pd.read_csv(stays_path)
# subjects = stays.subject_id.unique()
# break_up_diagnoses_by_subject(phenotypes, output_path, subjects=subjects)

#-----------------------------------------------------
#-----------------add category labels to fuu_info.csv
# import yaml
# import pandas as pd

# # 1. 加载 YAML 文件
# yaml_path = '/data/ke/MIMIC_subset/category2icd_code9_10.yaml'
# with open(yaml_path, 'r') as file:
#     yaml_data = yaml.safe_load(file)

# # 2. 筛选出 use_in_benchmark 为 true 的类别
# benchmark_classes = [key for key, value in yaml_data.items() if value.get('use_in_benchmark')]

# # 3. 读取 CSV 文件
# csv_path = '/data/ke/MIMIC_subset/hadmid_with_ECG_and_CXR_nonan_hasicd_no_decimal.csv'
# df = pd.read_csv(csv_path)

# # 4. 在 CSV 文件中添加新列
# for benchmark_class in benchmark_classes:
#     df[benchmark_class] = None  # 添加新列，初始值为 None

# # 5. 保存更新后的 CSV 文件
# df.to_csv(csv_path, index=False)

# import os
# import pandas as pd

# # 定义文件路径
# root_path = '/data/ke/MIMIC_subset/root/'
# diagnoses_path_template = os.path.join(root_path, '{subject_id}', 'diagnoses.csv')
# target_csv_path = '/data/ke/MIMIC_subset/hadmid_with_ECG_and_CXR_nonan_hasicd_no_decimal.csv'

# # 读取目标 CSV 文件
# df_target = pd.read_csv(target_csv_path)

# # 创建一个字典，用于存储每个 hadm_id 对应的 HCUP_CCS_2015 列
# hccp_map = {}

# # 遍历每个 subject_id 的目录
# for subject_id in os.listdir(root_path):
#     subject_folder = os.path.join(root_path, subject_id)
#     diagnoses_file = os.path.join(subject_folder, 'diagnoses.csv')
    
#     if os.path.isfile(diagnoses_file):
#         # 读取 diagnoses.csv 文件
#         df_diagnoses = pd.read_csv(diagnoses_file)
        
#         # 筛选 USE_IN_BENCHMARK 为 1.0 的行
#         df_benchmark = df_diagnoses[df_diagnoses['USE_IN_BENCHMARK'] == 1.0]
        
#         # 更新 hccp_map 字典
#         for _, row in df_benchmark.iterrows():
#             hadm_id = row['hadm_id']
#             hccp_code = row['HCUP_CCS_2015']
#             if hadm_id not in hccp_map:
#                 hccp_map[hadm_id] = set()
#             hccp_map[hadm_id].add(hccp_code)

# # 更新目标 CSV 文件
# for hadm_id, hccp_codes in hccp_map.items():
#     for hccp_code in hccp_codes:
#         if hccp_code not in df_target.columns:
#             df_target[hccp_code] = 0  # 添加新的列，并初始化为 0
    
#     # 对应的行更新为 1
#     if hadm_id in df_target['hadm_id'].values:
#         df_target.loc[df_target['hadm_id'] == hadm_id, list(hccp_codes)] = 1
# with_label_path='/data/ke/MIMIC_subset/with_label.csv'
# # 保存更新后的 CSV 文件
# df_target.to_csv(with_label_path, index=False)
#----------------------------------------------------------
#------choose PA in with_label.csv--------------------------
# import pandas as pd

# # 文件路径
# with_label_path = '/data/ke/MIMIC_subset/with_label.csv'
# metadata_path = '/data/ke/MedFuse/mimic4extract/data/MIMIC/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv'
# output_path = '/data/ke/MIMIC_subset/with_label_PA.csv'

# # 加载数据
# with_label_df = pd.read_csv(with_label_path)
# metadata_df = pd.read_csv(metadata_path)

# # 将first_CXR_studyid转化为整数形式（删除小数部分）
# with_label_df['first_CXR_studyid'] = with_label_df['first_CXR_studyid'].apply(lambda x: str(int(x)))

# # 将metadata中的study_id转为字符串以便合并
# metadata_df['study_id'] = metadata_df['study_id'].astype(str)

# # 提取metadata中ViewPosition为PA的行
# pa_metadata_df = metadata_df[metadata_df['ViewPosition'] == 'PA'][['study_id', 'ViewPosition', 'dicom_id']]

# # 合并with_label_df和pa_metadata_df，基于first_CXR_studyid和study_id
# filtered_df = pd.merge(with_label_df, pa_metadata_df,
#                        left_on='first_CXR_studyid', right_on='study_id', how='inner')

# # 删除不必要的列（如study_id，避免重复）
# filtered_df.drop(columns=['study_id'], inplace=True)
# filtered_df = filtered_df.drop_duplicates(subset=['first_CXR_studyid'])

# # 保存结果到with_label_PA.csv
# filtered_df.to_csv(output_path, index=False)

# print(f"Filtered data with PA ViewPosition saved to {output_path}")
#-------------------------------------------
#-------drop nolabel ones
# import pandas as pd

# # 文件路径
# input_path = '/data/ke/MIMIC_subset/with_label_PA.csv'
# output_path = '/data/ke/MIMIC_subset/with_nonan_label_PA.csv'

# # 需要筛选的列
# columns_to_check = [
#     'Chronic obstructive pulmonary disease and bronchiectasis',
#     'Congestive heart failure; nonhypertensive',
#     'Coronary atherosclerosis and other heart disease',
#     'Essential hypertension',
#     'Hypertension with complications and secondary hypertension',
#     'Other lower respiratory disease',
#     'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)'
# ]

# # 加载数据
# df = pd.read_csv(input_path)

# # 筛选出在指定列中至少有一个1的行
# filtered_df = df[df[columns_to_check].sum(axis=1) >= 1]

# # 保存结果到with_nonan_label_PA.csv
# filtered_df.to_csv(output_path, index=False)

# print(f"Filtered data saved to {output_path}")
#end of TODO for change data_path
