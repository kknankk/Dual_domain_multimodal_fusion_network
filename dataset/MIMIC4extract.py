import pandas as pd
import yaml

# Define file paths
admissions_file = "/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimiciv/3.0/hosp/admissions.csv"
edstays_file = "/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimic-iv-ed/2.2/ed/edstays.csv"
output_file = "/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/filtered_merged_data.csv"

# Load admissions and edstays CSV files
admissions_df = pd.read_csv(admissions_file)
edstays_df = pd.read_csv(edstays_file)

# Perform inner join on hadm_id
merged_df = pd.merge(admissions_df, edstays_df, on="hadm_id", how="inner")

# Select required columns
selected_columns = [
    "subject_id_x", "hadm_id", "admittime", "dischtime", "race_x",
    "edregtime", "edouttime", "stay_id", "intime", "outtime", "gender", "race_y"
]
filtered_df = merged_df[selected_columns]

# Save merged result
filtered_df.to_csv(output_file, index=False)
print(f"Merging and extraction completed, results saved to {output_file}")

# Read the merged data with labels
merged_labels = pd.read_csv(output_file)
merged_labels.rename(columns={'subject_id_x': 'subject_id'}, inplace=True)

# Convert time columns to datetime format
merged_labels['admittime'] = pd.to_datetime(merged_labels['admittime'])
merged_labels['dischtime'] = pd.to_datetime(merged_labels['dischtime'])
merged_labels['edregtime'] = pd.to_datetime(merged_labels['edregtime'])
merged_labels['endtime'] = merged_labels['edregtime'] + pd.Timedelta(hours=12)

# Read CXR metadata and process time fields
file_path = '/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv'
cxr_metadata = pd.read_csv(file_path)
cxr_metadata['StudyTime'] = cxr_metadata['StudyTime'].apply(lambda x: f'{int(float(x)):06}')
cxr_metadata['StudyDateTime'] = pd.to_datetime(cxr_metadata['StudyDate'].astype(str) + ' ' + cxr_metadata['StudyTime'].astype(str), format="%Y%m%d %H%M%S")
cxr_metadata = cxr_metadata.rename(columns={'StudyDateTime': 'CXRStudyDateTime'})
cxr_metadata1 = cxr_metadata[['dicom_id', 'subject_id', 'ViewPosition', 'CXRStudyDateTime']]

# Merge CXR metadata with label data
merged_df = pd.merge(cxr_metadata1, merged_labels, on='subject_id', how='inner')

# Filter CXR data based on CXRStudyDateTime between admittime and dischtime
filtered_df = merged_df[(merged_df['CXRStudyDateTime'] >= merged_df['edregtime']) & 
                        (merged_df['CXRStudyDateTime'] <= merged_df['endtime'])]
filtered_df = filtered_df[filtered_df['ViewPosition'].isin(['AP', 'PA'])]

# Process ECG data
record_list_df = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimic-iv-ecg/1.0/record_list.csv')
record_list_df['ecg_time'] = pd.to_datetime(record_list_df['ecg_time'])
record_list_df1 = record_list_df[['subject_id', 'ecg_time', 'path']]

# Merge ECG data with filtered CXR data
merged_df_ecg = pd.merge(record_list_df1, filtered_df, on='subject_id', how='inner')
merged_df_ecg = merged_df_ecg[(merged_df_ecg['ecg_time'] >= merged_df_ecg['edregtime']) & 
                              (merged_df_ecg['ecg_time'] <= merged_df_ecg['endtime'])]

# Save merged ECG data
output_file_path = '/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/merged1.csv'
merged_df_ecg.to_csv(output_file_path, index=False)
print(f"Filtered DataFrame saved to {output_file_path}")

# Read merged ECG data and extract necessary columns
input_file = '/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/merged1.csv'
output_file = '/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/merged2.csv'
columns_to_extract = [
    'subject_id', 'hadm_id', 'edregtime', 'endtime', 'ecg_time',
    'CXRStudyDateTime', 'path', 'dicom_id', 'ViewPosition',
    'admittime', 'dischtime'
]

# Extract columns and save
data = pd.read_csv(input_file)
filtered_data = data[columns_to_extract]
filtered_data.to_csv(output_file, index=False)

# Read deduplicated data and remove duplicates
input_file = '/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/merged2.csv'
output_file = '/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/merged2_deduplicated.csv'

data = pd.read_csv(input_file)
data_deduplicated = data.drop_duplicates()
data_deduplicated.to_csv(output_file, index=False)
print(f"Deduplicated data saved to {output_file}")

# Read the deduplicated data and group by hadm_id
file_path = '/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/merged2_deduplicated.csv'
data = pd.read_csv(file_path)
filtered_rows = []

for hadm_id, group in data.groupby('hadm_id'):
    if len(group) == 1:
        filtered_rows.append(group.iloc[0])
    else:
        unique_ecg_times = group['ecg_time'].drop_duplicates().tolist()
        if len(unique_ecg_times) > 1:
            second_ecg_time = unique_ecg_times[1]
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

result_df = pd.DataFrame(filtered_rows)
result_df.to_csv('/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/filtered_results.csv', index=False)
print("Processing completed, results saved to filtered_results.csv")

# Read ICD data
icd_file_path = '/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/physionet.org/files/mimiciv/3.0/hosp/diagnoses_icd.csv'
df_diagnoses = pd.read_csv(icd_file_path)

# Read YAML file
with open('/home/mimic/MIMIC_subset/MIMIC_subset/category2icd_code9_10.yaml', 'r') as file:
    category_data = yaml.safe_load(file)

# Extract categories and corresponding ICD codes
categories_to_include = {
    key: value['codes'] for key, value in category_data.items() if value['use_in_benchmark']
}

# Create label columns
label_columns = [
    'Congestive heart failure; nonhypertensive',
    'Coronary atherosclerosis and other heart disease',
    'Pneumonia_or_COPD_or_OLRD_167'
]

for category, codes in categories_to_include.items():
    df_diagnoses[category] = df_diagnoses['icd_code'].apply(lambda x: 1 if str(x) in codes else 0)

# Group by hadm_id
df_diagnoses = df_diagnoses.groupby('hadm_id', as_index=False).agg({**{label: 'max' for label in label_columns}})

# Read the filtered results
filtered_results = pd.read_csv('/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/filtered_results.csv')

# Filter diagnoses data for matching hadm_id
common_hadm_ids = filtered_results['hadm_id'].unique()
df_diagnoses_filtered = df_diagnoses[df_diagnoses['hadm_id'].isin(common_hadm_ids)]

# Extract relevant columns and merge
columns_to_extract = [
    'hadm_id', 'Congestive heart failure; nonhypertensive',
    'Coronary atherosclerosis and other heart disease',
    'Pneumonia_or_COPD_or_OLRD_167'
]
df_diagnoses_filtered = df_diagnoses_filtered[columns_to_extract]
final_output = pd.merge(filtered_results, df_diagnoses_filtered, on='hadm_id', how='inner')

# Save the final result
final_output.to_csv('/home/mimic/MIMIC_subset/MIMIC_subset/raw_database/final_results.csv', index=False)
print("Final results saved to final_results.csv")
