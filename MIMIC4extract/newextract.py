import pandas as pd
import yaml

# # 步骤 1：读取 YAML 文件并提取 use_in_benchmark 为 True 的 codes
# yaml_file_path = '/home/mimic/MIMIC_subset/MIMIC_subset/category2icd_code9_10.yaml'

# with open(yaml_file_path, 'r') as file:
#     data = yaml.safe_load(file)

# # 获取所有 use_in_benchmark 为 True 的 codes
# codes_to_extract = []

# for category, attributes in data.items():
#     if attributes.get('use_in_benchmark', False):
#         # print(f'category {category}')
#         # print(f'attributes {attributes}')
#         codes_to_extract.extend(attributes['codes'])

# # 将 codes 转为集合以去重
# codes_to_extract = set(codes_to_extract)

# # 步骤 2：读取 CSV 文件并提取对应的行
# csv_file_path = '/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimiciv/3.0/hosp/diagnoses_icd.csv'
# diagnoses_df = pd.read_csv(csv_file_path)

# # 过滤出 icd_code 在 codes_to_extract 中的行
# filtered_df = diagnoses_df[diagnoses_df['icd_code'].isin(codes_to_extract)]

# # 步骤 3：保存结果为新的 CSV 文件
# output_csv_path = '/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/7class_diagnoses.csv'
# filtered_df.to_csv(output_csv_path, index=False)

# print(f'Filtered data saved to {output_csv_path}')

# import pandas as pd

# # 文件路径
# diagnoses_file_path = '/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/7class_diagnoses.csv'
# admissions_file_path = '/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimiciv/3.0/icu/admissions.csv'

# # 步骤 1：读取 CSV 文件
# diagnoses_df = pd.read_csv(diagnoses_file_path)
# admissions_df = pd.read_csv(admissions_file_path)

# # 步骤 2：合并数据
# # 根据 hadm_id 进行合并，使用左连接以保留 diagnoses_df 中的所有行
# merged_df = pd.merge(diagnoses_df, admissions_df[['hadm_id', 'admittime', 'dischtime', 'deathtime']], on='hadm_id', how='left')

# # 步骤 3：保存结果为新的 CSV 文件
# output_file_path = '/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/7_diagnoses_with_time.csv'
# merged_df.to_csv(output_file_path, index=False)

# print(f'Merged data saved to {output_file_path}')


import pandas as pd
import yaml

# 读取 CSV 文件
# df_diagnoses = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/7_diagnoses_with_time.csv')

# # 读取 YAML 文件
# with open('/home/mimic/MIMIC_subset/MIMIC_subset/category2icd_code9_10.yaml', 'r') as file:
#     category_data = yaml.safe_load(file)

# # 提取符合条件的类别和对应的 ICD 代码
# categories_to_include = {
#     key: value['codes'] for key, value in category_data.items() if value['use_in_benchmark']
# }
# # print(f' categories_to_include {categories_to_include}')

# # 为每个类别添加新列
# for category, codes in categories_to_include.items():
#     print(f'category {category}')#疾病名
#     print(f'codes {codes}')#对应code
#     df_diagnoses[category] = df_diagnoses['icd_code'].apply(lambda x: 1 if str(x) in codes else 0)

# # 保存结果到新的 CSV 文件
# df_diagnoses.to_csv('/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/updated_diagnoses_with_time.csv', index=False)

# print("更新完成，新的 CSV 文件已保存。")

# import pandas as pd

# # 读取 CSV 文件
# df = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/updated_diagnoses_with_time.csv')

# # 定义标签列
# label_columns = [
#     'Chronic obstructive pulmonary disease and bronchiectasis',
#     'Congestive heart failure; nonhypertensive',
#     'Coronary atherosclerosis and other heart disease',
#     'Essential hypertension',
#     'Hypertension with complications and secondary hypertension',
#     'Other lower respiratory disease',
#     'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)'
# ]

# # 合并相同的 hadm_id 的标签，标签列使用按位或运算
# merged_df = df.groupby('hadm_id', as_index=False).agg({
#     'subject_id': 'first',
#     'seq_num': 'first',
#     'icd_code': 'first',
#     'icd_version': 'first',
#     'admittime': 'first',
#     'dischtime': 'first',
#     'deathtime': 'first',
#     **{label: 'max' for label in label_columns}
# })

# # 保存结果到新的 CSV 文件
# merged_df.to_csv('/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/merged_labels.csv', index=False)

# print("合并成功，新的 CSV 文件已保存。")
#===========================================之前都没问题

#===================下面是之前的
# import pandas as pd

# # 读取 CSV 文件
# record_list = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimic-iv-ecg/1.0/record_list.csv')
# merged_labels = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/merged_labels.csv')

# # 转换时间列为 datetime 格式
# merged_labels['admittime'] = pd.to_datetime(merged_labels['admittime'])
# merged_labels['dischtime'] = pd.to_datetime(merged_labels['dischtime'])
# merged_labels['deathtime'] = pd.to_datetime(merged_labels['deathtime'])
# a=merged_labels['admittime']
# # print(f'convert time for merged_labels.csv {a}')
# # 也将 ecg_time 转换为 datetime 格式
# record_list['ecg_time'] = pd.to_datetime(record_list['ecg_time'])
# b=record_list['ecg_time']
# # print(f'convert time for ecg.csv {b}')

# # 添加新的列，默认为 0
# # merged_labels['ecg_path'] = 0
# merged_labels['ecg_path'] = ''  # 初始化为字符串类型的空值


# # 遍历 merged_labels 的每一行
# for index, row in merged_labels.iterrows():
#     subject_id = row['subject_id']
#     admittime = row['admittime']
#     dischtime = row['dischtime']
#     deathtime = row['deathtime']
#     # dischtime_array = pd.to_datetime(dischtime)
#     # dischtime_array = pd.to_datetime(deathtime)

#     # 获取对应的 record_list 中的 ECG 记录
#     ecg_records = record_list[
#         (record_list['subject_id'] == subject_id) &
#         (record_list['ecg_time'] >= admittime) &
#         # (record_list['ecg_time'] <= (dischtime if pd.notnull(dischtime) else deathtime))
#         (record_list['ecg_time'] <= (dischtime))
#         # (record_list['ecg_time'] <= (dischtime_array)|record_list['ecg_time'] <= (dischtime_array))
#     ]
#     # print(f'-------------{ecg_records}---------------')
#     if not ecg_records.empty:
#         # merged_labels.at[index, 'ecg_path'] = ecg_records.iloc[0]['path']
#         random_record = ecg_records.sample(n=1) 

#         merged_labels.at[index, 'ecg_path'] = random_record.iloc[0]['path']

# # 可选：保存修改后的 DataFrame
# merged_labels.to_csv('/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/with_random_ecg.csv', index=False)
#==========ecg完成
#==========提取cxr==================================
# import pandas as pd

# # 读取 CSV 文件
# cxr_metadata = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv')
# merged_metadata = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/with_random_ecg.csv')

# cxr_metadata['StudyTime'] = cxr_metadata['StudyTime'].apply(lambda x: f'{int(float(x)):06}' )
# cxr_metadata['StudyDateTime'] = pd.to_datetime(cxr_metadata['StudyDate'].astype(str) + ' ' + cxr_metadata['StudyTime'].astype(str) ,format="%Y%m%d %H%M%S")

# columns = ['ViewPosition', 'dicom_id', 'subject_id', 'StudyDateTime']

# # only common subjects with both icu stay and an xray
# cxr_merged_data = merged_metadata.merge(cxr_metadata[columns], how='inner', on='subject_id')



# cxr_merged_data['intime'] = pd.to_datetime(cxr_merged_data['admittime'])
# cxr_merged_data['outtime'] = pd.to_datetime(cxr_merged_data['dischtime'])
# end_time = cxr_merged_data['dischtime']
# cxr_merged_icustays_during = cxr_merged_data.loc[(cxr_merged_data.StudyDateTime>=cxr_merged_data['intime'])&((cxr_merged_data.StudyDateTime<=end_time))]

# # cxr_merged_icustays_AP = cxr_merged_icustays_during[cxr_merged_icustays_during['ViewPosition'] == 'PA']
# cxr_merged_icustays_AP = cxr_merged_icustays_during[cxr_merged_icustays_during['ViewPosition'].isin(['PA', 'AP'])]

# groups = cxr_merged_icustays_AP.groupby('hadm_id')

# groups_selected = []
# for group in groups:
#     # select the latest cxr for the icu stay
#     selected = group[1].sort_values('StudyDateTime').head(1).reset_index()
#     groups_selected.append(selected)
# groups = pd.concat(groups_selected, ignore_index=True)
# # print(f'groups {groups}')
# groups.to_csv('/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/with_2side_cxr&random_ecg.csv', index=False)


# import pandas as pd

# # 读取 CSV 文件
# file_path = '/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/with_2side_cxr&random_ecg.csv'
# data = pd.read_csv(file_path)
# # data =groups
# # 提取 ecg_path 和 dicom_id 都不为空的行
# filtered_data = data[data['ecg_path'].notnull() & (data['ecg_path'] != '') & 
#                      data['dicom_id'].notnull() & (data['dicom_id'] != '')]

# # 保存到新的 CSV 文件
# output_file_path = '/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/final_3_0.csv'
# filtered_data.to_csv(output_file_path)

# print(f"提取的行已保存到 {output_file_path}")



#===============试着删去和合并label===================
import pandas as pd

# 读取 CSV 文件
file_path = '/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/new2.csv'
data = pd.read_csv(file_path)

# 合并指定列，取较大的值
# data['Pneumonia_or_COPD_or_OLRD_167'] = data[['Pneumonia (except that caused by tuberculosis or sexually transmitted disease)', 
#                                     'Chronic obstructive pulmonary disease and bronchiectasis','Other lower respiratory disease']].max(axis=1)
# data['EH_OR_HWCASH_45'] = data[['Essential hypertension', 
#                                     'Hypertension with complications and secondary hypertension']].max(axis=1)

# 找到 'Other lower respiratory disease' 列的索引
b = data.columns.get_loc('Pneumonia_or_COPD_or_OLRD_167')

# 重新排列列，将新列插入到 'Other lower respiratory disease' 列后面
columns = list(data.columns)
columns.insert(b + 1, columns.pop(columns.index('EH_OR_HWCASH_45')))
data = data[columns]

# 可以选择删除原始的两列，如果不需要的话
# data.drop(columns=['Pneumonia (except that caused by tuberculosis or sexually transmitted disease)', 
#                                     'Chronic obstructive pulmonary disease and bronchiectasis','Other lower respiratory disease','Essential hypertension', 
#                                     'Hypertension with complications and secondary hypertension'], inplace=True)

out_path = '/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/new3.csv'

# 将结果保存回 CSV 文件
data.to_csv(out_path, index=False)

print("合并完成，新列已插入到指定位置，结果已保存到 CSV 文件中。")


# import pandas as pd

# # 读取 CSV 文件
# file_path = '/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/test.csv'
# data = pd.read_csv(file_path)

# # 定义要筛选的列
# columns_to_check = [
#     'Congestive heart failure; nonhypertensive',
#     'Coronary atherosclerosis and other heart disease',
#     'Hypertension with complications and secondary hypertension',
#     'Pneumonia_or_COPD'
# ]

# # 筛选至少有一列不为 0 的行
# filtered_data = data[data[columns_to_check].ne(0).any(axis=1)]

# # 保存结果到新的 CSV 文件
# output_file_path = '/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/test1.csv'
# filtered_data.to_csv(output_file_path, index=False)

# print(f"筛选完成，结果已保存到 {output_file_path}。")






















































# import pdb; pdb.set_trace()

# groups['cxr_length'] = (groups['StudyDateTime'] - groups['intime']).astype('timedelta64[h]')






# import pandas as pd

# # 加载数据
# merged_labels_path = '/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/updated_merged_labels.csv'
# metadata_path = '/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv'

# # 读取 CSV 文件
# updated_labels_df = pd.read_csv(merged_labels_path)
# metadata_df = pd.read_csv(metadata_path)

# # 将 admittime, dischtime, deathtime 转换为 datetime
# updated_labels_df['admittime'] = pd.to_datetime(updated_labels_df['admittime'])
# updated_labels_df['dischtime'] = pd.to_datetime(updated_labels_df['dischtime'])
# updated_labels_df['deathtime'] = pd.to_datetime(updated_labels_df['deathtime'])

# # 提取 StudyDate 的日期格式并转换为 datetime
# metadata_df['StudyDate'] = pd.to_datetime(metadata_df['StudyDate'].astype(str), format='%Y%m%d')

# # 创建一个新的列来存储条件结果
# def check_conditions(row):
#     # 获取当前 subject_id
#     subject_id = row['subject_id']
#     # 获取 admittime, dischtime 和 deathtime
#     admittime = row['admittime']
#     dischtime = row['dischtime']
#     deathtime = row['deathtime']
    
#     # 查找 metadata_df 中符合条件的行
#     matching_records = metadata_df[(metadata_df['subject_id'] == subject_id) &
#                                    (metadata_df['ViewPosition'] == 'PA') &
#                                    (metadata_df['StudyDate'].between(admittime, dischtime) |
#                                     metadata_df['StudyDate'].between(admittime, deathtime))]
    
#     # 返回 1 或 0
#     return 1 if not matching_records.empty else 0

# # 应用条件检查
# updated_labels_df['new_column'] = updated_labels_df.apply(check_conditions, axis=1)

# # 保存结果到新的 CSV 文件
# output_path = '/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/updated_labels_with_conditions.csv'
# updated_labels_df.to_csv(output_path, index=False)

# print(f"处理完成，结果已保存到 {output_path}")

# import pandas as pd

# # 加载数据
# merged_labels_path = '/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/updated_merged_labels.csv'
# metadata_path = '/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv'

# # 读取 CSV 文件
# updated_labels_df = pd.read_csv(merged_labels_path)
# metadata_df = pd.read_csv(metadata_path)

# # 将 admittime, dischtime, deathtime 转换为 datetime
# updated_labels_df['admittime'] = pd.to_datetime(updated_labels_df['admittime'])
# updated_labels_df['dischtime'] = pd.to_datetime(updated_labels_df['dischtime'])
# updated_labels_df['deathtime'] = pd.to_datetime(updated_labels_df['deathtime'])

# # 提取 StudyDate 的日期格式并转换为 datetime
# metadata_df['StudyDate'] = pd.to_datetime(metadata_df['StudyDate'].astype(str), format='%Y%m%d')

# # 筛选出 ViewPosition 为 PA 的记录
# metadata_filtered = metadata_df[metadata_df['ViewPosition'] == 'PA']

# # 合并两个 DataFrame
# merged_df = updated_labels_df.merge(metadata_filtered[['subject_id', 'StudyDate']], on='subject_id', how='left')

# # 创建条件列
# merged_df['in_time_range'] = (
#     (merged_df['StudyDate'] >= merged_df['admittime']) & 
#     (merged_df['StudyDate'] <= merged_df['dischtime']) | 
#     (merged_df['StudyDate'] >= merged_df['admittime']) & 
#     (merged_df['StudyDate'] <= merged_df['deathtime'])
# )

# # 将布尔值转换为 1 和 0
# merged_df['new_column'] = merged_df['in_time_range'].astype(int)

# # 只保留原始列和新列
# result_df = merged_df[updated_labels_df.columns.tolist() + ['new_column']]

# # 保存结果到新的 CSV 文件
# output_path = '/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/updated_labels_with_conditions.csv'
# result_df.to_csv(output_path, index=False)

# print(f"处理完成，结果已保存到 {output_path}")

# import pandas as pd

# # 读取 CSV 文件
# test_df = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/test.csv')
# label_df = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/with_nonan_label_PA.csv')

# # 提取 hadm_id 列
# test_hadm_ids = set(test_df['hadm_id'])
# label_hadm_ids = set(label_df['hadm_id'])

# # 找到重叠的 hadm_id
# common_hadm_ids = test_hadm_ids.intersection(label_hadm_ids)

# # 输出重叠的数量
# print(f'Number of hadm_id in both files: {len(common_hadm_ids)}')

# import pandas as pd

# # 读取 CSV 文件
# test_df = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/test.csv')

# # 只保留 ecg_exists 为 1 的行
# filtered_df = test_df[test_df['ecg_exists'] == 1]

# # 将结果写入原 CSV 文件
# filtered_df.to_csv('/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/final1.csv', index=False)

# import pandas as pd

# # 读取 CSV 文件
# final_df = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/final1.csv')
# label_df = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/with_nonan_label_PA.csv')
# all_df=pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/7_diagnoses_with_time.csv')
# # 提取 hadm_id 列
# all_hadm_ids = set(all_df['hadm_id'])
# label_hadm_ids = set(label_df['hadm_id'])

# # 找到一致的 hadm_id
# common_hadm_ids = all_hadm_ids.intersection(label_hadm_ids)

# # 输出一致的数量
# print(f'Number of matching hadm_id: {len(common_hadm_ids)}')


#===检查
# import pandas as pd

# # 读取两个 CSV 文件
# updated_labels_df = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/updated_merged_labels.csv')
# label_df = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/with_nonan_label_PA.csv')

# # 将 NaN 替换为 0
# updated_labels_df.fillna(0, inplace=True)
# label_df.fillna(0, inplace=True)

# # 获取两个 DataFrame 中的 hadm_id 列
# updated_hadm_ids = set(updated_labels_df['hadm_id'])
# label_hadm_ids = set(label_df['hadm_id'])

# # 找到两个 DataFrame 中相同的 hadm_id
# common_hadm_ids = updated_hadm_ids.intersection(label_hadm_ids)

# # 存储不一致的 hadm_id
# inconsistent_hadm_ids = []

# # 检查相同 hadm_id 的七列疾病数据是否一致
# disease_columns = [
#     'Chronic obstructive pulmonary disease and bronchiectasis',
#     'Congestive heart failure; nonhypertensive',
#     'Coronary atherosclerosis and other heart disease',
#     'Essential hypertension',
#     'Hypertension with complications and secondary hypertension',
#     'Other lower respiratory disease',
#     'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)'
# ]

# for hadm_id in common_hadm_ids:
#     # 获取对应的行
#     updated_row = updated_labels_df[updated_labels_df['hadm_id'] == hadm_id][disease_columns].values
#     label_row = label_df[label_df['hadm_id'] == hadm_id][disease_columns].values
    
#     # 检查是否存在不一致
#     if (updated_row != label_row).any():
#         inconsistent_hadm_ids.append(hadm_id)

# # 输出不一致的 hadm_id
# if inconsistent_hadm_ids:
#     print("不一致的 hadm_id:")
#     print(inconsistent_hadm_ids)
# else:
#     print("所有相同的 hadm_id 的疾病数据一致。")
