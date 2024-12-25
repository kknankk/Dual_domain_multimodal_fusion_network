#-----------------1--------------------
import pandas as pd


admissions_file = "/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimiciv/3.0/hosp/admissions.csv"
edstays_file = "/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimic-iv-ed/2.2/ed/edstays.csv"
output_file = "/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/filtered_merged_data.csv"


admissions_df = pd.read_csv(admissions_file)
edstays_df = pd.read_csv(edstays_file)


merged_df = pd.merge(admissions_df, edstays_df, on="hadm_id", how="inner")


selected_columns = [
    "subject_id_x", "hadm_id", "admittime", "dischtime", "race_x",
    "edregtime", "edouttime", "stay_id", "intime", "outtime", "gender", "race_y"
]
merged_labels = merged_df[selected_columns]



#----------------1-------------------------


#------------------2----------------------------

merged_labels.rename(columns={'subject_id_x': 'subject_id'}, inplace=True)

merged_labels['admittime'] = pd.to_datetime(merged_labels['admittime'])
merged_labels['dischtime'] = pd.to_datetime(merged_labels['dischtime'])
merged_labels['edregtime'] = pd.to_datetime(merged_labels['edregtime'])
merged_labels['endtime'] = merged_labels['edregtime'] + pd.Timedelta(hours=12)


file_path = '/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv'
cxr_metadata = pd.read_csv(file_path)
cxr_metadata['StudyTime'] = cxr_metadata['StudyTime'].apply(lambda x: f'{int(float(x)):06}' )
cxr_metadata['StudyDateTime'] = pd.to_datetime(cxr_metadata['StudyDate'].astype(str) + ' ' + cxr_metadata['StudyTime'].astype(str) ,format="%Y%m%d %H%M%S")
# print(cxr_metadata)
cxr_metadata = cxr_metadata.rename(columns={'StudyDateTime': 'CXRStudyDateTime'})
cxr_metadata1=cxr_metadata[['dicom_id', 'subject_id', 'ViewPosition', 'CXRStudyDateTime']]




merged_cxr = pd.merge(cxr_metadata1, merged_labels, on='subject_id', how='inner')

# # Filter the rows where StudyDateTime is between admittime and dischtime
filtered_cxr = merged_cxr[(merged_cxr['CXRStudyDateTime'] >= merged_cxr['edregtime']) & 
                        (merged_cxr['CXRStudyDateTime'] <= merged_cxr['endtime'])]
filtered_cxr = filtered_cxr[filtered_cxr['ViewPosition'].isin(['AP', 'PA'])]


# #---------ecg------------
record_list_df = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimic-iv-ecg/1.0/record_list.csv')
# # record_list_df = pd.read_csv(record_list)
record_list_df['ecg_time'] = pd.to_datetime(record_list_df['ecg_time'])

# # print(record_list_df.head())
record_list_df1=record_list_df[['subject_id','ecg_time','path']]
merged_df_ecg = pd.merge(record_list_df1, filtered_cxr, on='subject_id', how='inner')
merged_df_ecg = merged_df_ecg[(merged_df_ecg['ecg_time'] >= merged_df_ecg['edregtime']) & 
                        (merged_df_ecg['ecg_time'] <= merged_df_ecg['endtime'])]



#---------------------2----------------------

#----------2.1----------------------



columns_to_extract = [
    'subject_id', 'hadm_id', 'edregtime', 'endtime', 'ecg_time',
    'CXRStudyDateTime', 'path', 'dicom_id', 'ViewPosition',
    'admittime', 'dischtime'
]


# data = pd.read_csv(input_file)
filtered_data = merged_df_ecg[columns_to_extract]




data_deduplicated = filtered_data.drop_duplicates()


#-----------2.1------------------

#------------2.2----------------

filtered_rows = []


for hadm_id, group in data_deduplicated.groupby('hadm_id'):
    if len(group) == 1:
        filtered_rows.append(group.iloc[0])
    else:
        unique_ecg_times = group['ecg_time'].drop_duplicates().tolist()
        
        if len(unique_ecg_times) > 1:
            second_ecg_time = unique_ecg_times[1]  # second ecg_time
            second_ecg_time_rows = group[group['ecg_time'] == second_ecg_time]
            
            if len(second_ecg_time_rows) > 1:
                unique_cxr_times = second_ecg_time_rows['CXRStudyDateTime'].drop_duplicates().tolist()
                if len(unique_cxr_times) > 1:
                    second_cxr_time = unique_cxr_times[1]
                    final_row = second_ecg_time_rows[second_ecg_time_rows['CXRStudyDateTime'] == second_cxr_time].iloc[0]
                else:
                    final_row = second_ecg_time_rows.iloc[0]
            else:
                final_row = second_ecg_time_rows.iloc[0]
            
            filtered_rows.append(final_row)
        else:
            continue


result_df = pd.DataFrame(filtered_rows)


#---------------2.2--------------
#---------------3----------------------
import yaml

icd_file_path = '/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimiciv/3.0/hosp/diagnoses_icd.csv'
df_diagnoses = pd.read_csv(icd_file_path)


with open('/home/mimic/MIMIC_subset/MIMIC_subset/category2icd_code9_10.yaml', 'r') as file:
    category_data = yaml.safe_load(file)


categories_to_include = {
    key: value['codes'] for key, value in category_data.items() if value['use_in_benchmark']
}


label_columns = [
    'Congestive heart failure; nonhypertensive',
    'Coronary atherosclerosis and other heart disease',
    'Pneumonia_or_COPD_or_OLRD_167'
]

# Create binary columns for the specified categories
for category, codes in categories_to_include.items():
    df_diagnoses[category] = df_diagnoses['icd_code'].apply(lambda x: 1 if str(x) in codes else 0)
# print(f'df_diagnoses {df_diagnoses.head()}')

df_diagnoses = df_diagnoses.groupby('hadm_id', as_index=False).agg({

    **{label: 'max' for label in label_columns}
})
# print(f'df_diagnoses {df_diagnoses.head()}')

# #------------




save_path = '/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/filtered_results_label.csv'


common_hadm_ids = result_df['hadm_id'].unique()
df_diagnoses_filtered = df_diagnoses[df_diagnoses['hadm_id'].isin(common_hadm_ids)]


columns_to_extract = [
    'hadm_id',
    'Congestive heart failure; nonhypertensive',
    'Coronary atherosclerosis and other heart disease',
    'Pneumonia_or_COPD_or_OLRD_167'
]
df_diagnoses_filtered = df_diagnoses_filtered[columns_to_extract]


merged_results = pd.merge(
    result_df,
    df_diagnoses_filtered,
    on='hadm_id',
    how='inner'
)


merged_results.to_csv(save_path, index=False)
# print(f"done, save path: {save_path}")


#--------------3-----------------------

#----------4.1 add troponin T
# import pandas as pd


a_df = pd.read_csv(save_path, parse_dates=['edregtime', 'endtime'])
b_df = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimiciv/3.0/hosp/labevents.csv', parse_dates=['charttime'])


merged_df = pd.merge(b_df, a_df, on='subject_id', how='inner')

#fliter record that 'charttime' is between 'admittime' and 'dischtime'
filtered_df = merged_df[
    (merged_df['charttime'] >= merged_df['edregtime']) &
    (merged_df['charttime'] <= merged_df['endtime'])
]


itemid_51002_df = filtered_df[filtered_df['itemid'] == 51003]



first_records_df = itemid_51002_df.groupby('subject_id').first().reset_index()


a_df = pd.merge(a_df, first_records_df[['subject_id', 'valuenum']], on='subject_id', how='left')


a_df = a_df.rename(columns={'valuenum': 'itemid_51003_valuenum'})


itemid_50908_df = filtered_df[filtered_df['itemid'] == 50908]
first_records_df_50908 = itemid_50908_df.groupby('subject_id').first().reset_index()
a_df = pd.merge(a_df, first_records_df_50908[['subject_id', 'valuenum']], on='subject_id', how='left')
a_df = a_df.rename(columns={'valuenum': 'itemid_50908_valuenum'})
# print(a_df.head())
itemid_50963_df = filtered_df[filtered_df['itemid'] == 50963]

first_records_df_50963 = itemid_50963_df.groupby('subject_id').first().reset_index()


a_df = pd.merge(a_df, first_records_df_50963[['subject_id', 'valuenum']], on='subject_id', how='left')
a_df = a_df.rename(columns={'valuenum': 'itemid_50963_valuenum'})
# a_df.to_csv('/home/mimic/MIMIC_subset/MIMIC_subset/PA_subset/with_nonan_label_PA_val.csv', index=False)

# print("saved at /home/mimic/MIMIC_subset/MIMIC_subset/PA_subset/with_nonan_label_PA_val.csv")
#----------------4.1


b_df = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimiciv/3.0/hosp/labevents.csv', parse_dates=['charttime'])


merged_df = pd.merge(b_df, a_df, on='subject_id', how='inner')


filtered_df = merged_df[
    (merged_df['charttime'] >= merged_df['edregtime']) &
    (merged_df['charttime'] <= merged_df['endtime'])
]


itemid_50963_df = filtered_df[filtered_df['itemid'] == 50963]
first_records_df_50963 = itemid_50963_df.groupby('subject_id').first().reset_index()


if not first_records_df_50963.empty:
    normalized_values = []
        # normalized_values = []
    subject_ids = []
    for _, row in first_records_df_50963.iterrows():
        ref_lower = row['ref_range_lower']
        ref_upper = row['ref_range_upper']
        value = row['valuenum']
        

        if pd.notnull(ref_lower) and pd.notnull(ref_upper) and ref_upper > ref_lower:
      
            normalized_value = (value - ref_lower) / (ref_upper - ref_lower)
      
            # normalized_value = max(0, min(1, normalized_value))
            normalized_values.append(normalized_value)
            subject_ids.append(row['subject_id'])


    normalized_df = pd.DataFrame({'subject_id': subject_ids, 'itemid_50963_valuenum': normalized_values})

 
    a_df = a_df.merge(normalized_df, on='subject_id', how='left', suffixes=('', '_new'))
    a_df['itemid_50963_valuenum'] = a_df['itemid_50963_valuenum_new'].fillna(a_df['itemid_50963_valuenum'])
    a_df.drop(columns=['itemid_50963_valuenum_new'], inplace=True)

#----------------5 add vital siganl

# import pandas as pd

b_df = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimic-iv-ed/2.2/ed/vitalsign.csv', parse_dates=['charttime'])


merged_df = pd.merge(b_df, a_df, on='subject_id', how='inner')


filtered_df = merged_df[
    (merged_df['charttime'] >= merged_df['edregtime']) & 
    (merged_df['charttime'] <= merged_df['endtime'])
]


first_records_df = filtered_df.groupby('subject_id').first().reset_index()


first_records_df = first_records_df[['subject_id', 'temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp']]


a_df = pd.merge(a_df, first_records_df, on='subject_id', how='left')


a_df.to_csv('/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/finnal_add_clinical.csv', index=False)
