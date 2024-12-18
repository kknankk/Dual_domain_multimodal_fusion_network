import pandas as pd
import yaml


# yaml_file_path = '/home/mimic/MIMIC_subset/MIMIC_subset/category2icd_code9_10.yaml'

# with open(yaml_file_path, 'r') as file:
#     data = yaml.safe_load(file)


# codes_to_extract = []

# for category, attributes in data.items():
#     if attributes.get('use_in_benchmark', False):
#         # print(f'category {category}')
#         # print(f'attributes {attributes}')
#         codes_to_extract.extend(attributes['codes'])


# codes_to_extract = set(codes_to_extract)


# csv_file_path = '/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimiciv/3.0/hosp/diagnoses_icd.csv'
# diagnoses_df = pd.read_csv(csv_file_path)


# filtered_df = diagnoses_df[diagnoses_df['icd_code'].isin(codes_to_extract)]


# output_csv_path = '/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/7class_diagnoses.csv'
# filtered_df.to_csv(output_csv_path, index=False)

# print(f'Filtered data saved to {output_csv_path}')

# import pandas as pd


# diagnoses_file_path = '/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/7class_diagnoses.csv'
# admissions_file_path = '/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimiciv/3.0/icu/admissions.csv'


# diagnoses_df = pd.read_csv(diagnoses_file_path)
# admissions_df = pd.read_csv(admissions_file_path)


# merged_df = pd.merge(diagnoses_df, admissions_df[['hadm_id', 'admittime', 'dischtime', 'deathtime']], on='hadm_id', how='left')


# output_file_path = '/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/7_diagnoses_with_time.csv'
# merged_df.to_csv(output_file_path, index=False)

# print(f'Merged data saved to {output_file_path}')


import pandas as pd
import yaml


# df_diagnoses = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/7_diagnoses_with_time.csv')


# with open('/home/mimic/MIMIC_subset/MIMIC_subset/category2icd_code9_10.yaml', 'r') as file:
#     category_data = yaml.safe_load(file)


# categories_to_include = {
#     key: value['codes'] for key, value in category_data.items() if value['use_in_benchmark']
# }
# # print(f' categories_to_include {categories_to_include}')


# for category, codes in categories_to_include.items():
#     print(f'category {category}')
#     print(f'codes {codes}')
#     df_diagnoses[category] = df_diagnoses['icd_code'].apply(lambda x: 1 if str(x) in codes else 0)


# df_diagnoses.to_csv('/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/updated_diagnoses_with_time.csv', index=False)



# import pandas as pd


# df = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/updated_diagnoses_with_time.csv')


# label_columns = [
#     'Chronic obstructive pulmonary disease and bronchiectasis',
#     'Congestive heart failure; nonhypertensive',
#     'Coronary atherosclerosis and other heart disease',
#     'Essential hypertension',
#     'Hypertension with complications and secondary hypertension',
#     'Other lower respiratory disease',
#     'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)'
# ]


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


# merged_df.to_csv('/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/merged_labels.csv', index=False)


# import pandas as pd


# record_list = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimic-iv-ecg/1.0/record_list.csv')
# merged_labels = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/merged_labels.csv')


# merged_labels['admittime'] = pd.to_datetime(merged_labels['admittime'])
# merged_labels['dischtime'] = pd.to_datetime(merged_labels['dischtime'])
# merged_labels['deathtime'] = pd.to_datetime(merged_labels['deathtime'])
# a=merged_labels['admittime']
# # print(f'convert time for merged_labels.csv {a}')

# record_list['ecg_time'] = pd.to_datetime(record_list['ecg_time'])
# b=record_list['ecg_time']
# # print(f'convert time for ecg.csv {b}')


# # merged_labels['ecg_path'] = 0
# merged_labels['ecg_path'] = '' 



# for index, row in merged_labels.iterrows():
#     subject_id = row['subject_id']
#     admittime = row['admittime']
#     dischtime = row['dischtime']
#     deathtime = row['deathtime']
#     # dischtime_array = pd.to_datetime(dischtime)
#     # dischtime_array = pd.to_datetime(deathtime)


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


# merged_labels.to_csv('/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/with_random_ecg.csv', index=False)

# import pandas as pd

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


# file_path = '/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/with_2side_cxr&random_ecg.csv'
# data = pd.read_csv(file_path)
# # data =groups

# filtered_data = data[data['ecg_path'].notnull() & (data['ecg_path'] != '') & 
#                      data['dicom_id'].notnull() & (data['dicom_id'] != '')]


# output_file_path = '/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/final_3_0.csv'
# filtered_data.to_csv(output_file_path)




#==============
import pandas as pd


file_path = '/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/new2.csv'
data = pd.read_csv(file_path)


# data['Pneumonia_or_COPD_or_OLRD_167'] = data[['Pneumonia (except that caused by tuberculosis or sexually transmitted disease)', 
#                                     'Chronic obstructive pulmonary disease and bronchiectasis','Other lower respiratory disease']].max(axis=1)
# data['EH_OR_HWCASH_45'] = data[['Essential hypertension', 
#                                     'Hypertension with complications and secondary hypertension']].max(axis=1)

b = data.columns.get_loc('Pneumonia_or_COPD_or_OLRD_167')


columns = list(data.columns)
columns.insert(b + 1, columns.pop(columns.index('EH_OR_HWCASH_45')))
data = data[columns]


# data.drop(columns=['Pneumonia (except that caused by tuberculosis or sexually transmitted disease)', 
#                                     'Chronic obstructive pulmonary disease and bronchiectasis','Other lower respiratory disease','Essential hypertension', 
#                                     'Hypertension with complications and secondary hypertension'], inplace=True)

out_path = '/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/new3.csv'


data.to_csv(out_path, index=False)




# import pandas as pd


# file_path = '/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/test.csv'
# data = pd.read_csv(file_path)


# columns_to_check = [
#     'Congestive heart failure; nonhypertensive',
#     'Coronary atherosclerosis and other heart disease',
#     'Hypertension with complications and secondary hypertension',
#     'Pneumonia_or_COPD'
# ]


# filtered_data = data[data[columns_to_check].ne(0).any(axis=1)]


# output_file_path = '/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/test1.csv'
# filtered_data.to_csv(output_file_path, index=False)
























































# import pdb; pdb.set_trace()

# groups['cxr_length'] = (groups['StudyDateTime'] - groups['intime']).astype('timedelta64[h]')






# import pandas as pd


# merged_labels_path = '/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/updated_merged_labels.csv'
# metadata_path = '/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv'


# updated_labels_df = pd.read_csv(merged_labels_path)
# metadata_df = pd.read_csv(metadata_path)


# updated_labels_df['admittime'] = pd.to_datetime(updated_labels_df['admittime'])
# updated_labels_df['dischtime'] = pd.to_datetime(updated_labels_df['dischtime'])
# updated_labels_df['deathtime'] = pd.to_datetime(updated_labels_df['deathtime'])


# metadata_df['StudyDate'] = pd.to_datetime(metadata_df['StudyDate'].astype(str), format='%Y%m%d')


# def check_conditions(row):

#     subject_id = row['subject_id']

#     admittime = row['admittime']
#     dischtime = row['dischtime']
#     deathtime = row['deathtime']
    

#     matching_records = metadata_df[(metadata_df['subject_id'] == subject_id) &
#                                    (metadata_df['ViewPosition'] == 'PA') &
#                                    (metadata_df['StudyDate'].between(admittime, dischtime) |
#                                     metadata_df['StudyDate'].between(admittime, deathtime))]
    

#     return 1 if not matching_records.empty else 0


# updated_labels_df['new_column'] = updated_labels_df.apply(check_conditions, axis=1)


# output_path = '/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/updated_labels_with_conditions.csv'
# updated_labels_df.to_csv(output_path, index=False)



# import pandas as pd


# merged_labels_path = '/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/updated_merged_labels.csv'
# metadata_path = '/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv'


# updated_labels_df = pd.read_csv(merged_labels_path)
# metadata_df = pd.read_csv(metadata_path)


# updated_labels_df['admittime'] = pd.to_datetime(updated_labels_df['admittime'])
# updated_labels_df['dischtime'] = pd.to_datetime(updated_labels_df['dischtime'])
# updated_labels_df['deathtime'] = pd.to_datetime(updated_labels_df['deathtime'])


# metadata_df['StudyDate'] = pd.to_datetime(metadata_df['StudyDate'].astype(str), format='%Y%m%d')


# metadata_filtered = metadata_df[metadata_df['ViewPosition'] == 'PA']


# merged_df = updated_labels_df.merge(metadata_filtered[['subject_id', 'StudyDate']], on='subject_id', how='left')


# merged_df['in_time_range'] = (
#     (merged_df['StudyDate'] >= merged_df['admittime']) & 
#     (merged_df['StudyDate'] <= merged_df['dischtime']) | 
#     (merged_df['StudyDate'] >= merged_df['admittime']) & 
#     (merged_df['StudyDate'] <= merged_df['deathtime'])
# )


# merged_df['new_column'] = merged_df['in_time_range'].astype(int)


# result_df = merged_df[updated_labels_df.columns.tolist() + ['new_column']]


# output_path = '/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/updated_labels_with_conditions.csv'
# result_df.to_csv(output_path, index=False)



# import pandas as pd


# test_df = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/test.csv')
# label_df = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/with_nonan_label_PA.csv')


# test_hadm_ids = set(test_df['hadm_id'])
# label_hadm_ids = set(label_df['hadm_id'])


# common_hadm_ids = test_hadm_ids.intersection(label_hadm_ids)


# print(f'Number of hadm_id in both files: {len(common_hadm_ids)}')

# import pandas as pd


# test_df = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/test.csv')


# filtered_df = test_df[test_df['ecg_exists'] == 1]


# filtered_df.to_csv('/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/final1.csv', index=False)

# import pandas as pd

# final_df = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/final1.csv')
# label_df = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/with_nonan_label_PA.csv')
# all_df=pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/7_diagnoses_with_time.csv')

# all_hadm_ids = set(all_df['hadm_id'])
# label_hadm_ids = set(label_df['hadm_id'])


# common_hadm_ids = all_hadm_ids.intersection(label_hadm_ids)


# print(f'Number of matching hadm_id: {len(common_hadm_ids)}')



# import pandas as pd


# updated_labels_df = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/MIMIC4extract/updated_merged_labels.csv')
# label_df = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/with_nonan_label_PA.csv')

# updated_labels_df.fillna(0, inplace=True)
# label_df.fillna(0, inplace=True)


# updated_hadm_ids = set(updated_labels_df['hadm_id'])
# label_hadm_ids = set(label_df['hadm_id'])


# common_hadm_ids = updated_hadm_ids.intersection(label_hadm_ids)


# inconsistent_hadm_ids = []


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
#  
#     updated_row = updated_labels_df[updated_labels_df['hadm_id'] == hadm_id][disease_columns].values
#     label_row = label_df[label_df['hadm_id'] == hadm_id][disease_columns].values
    
#   
#     if (updated_row != label_row).any():
#         inconsistent_hadm_ids.append(hadm_id)


