import pandas as pd
from sklearn.model_selection import train_test_split
import os

# File paths
input_csv_path = '/home/mimic/MIMIC_subset/MIMIC_subset/PA_subset/with_nonan_label_PA_val.csv'
val_folder = '/home/mimic/MIMIC_subset/MIMIC_subset/PA_subset'
test_folder = '/home/mimic/MIMIC_subset/MIMIC_subset/PA_subset'

# Load the data
data = pd.read_csv(input_csv_path)

# Define the disease columns
diseases = [
    'Congestive heart failure; nonhypertensive',
    'Coronary atherosclerosis and other heart disease',
    'Pneumonia_or_COPD_or_OLRD_167'
]

# Fill missing values
data[diseases] = data[diseases].fillna(0)

# Create combined label column, ensuring each label combination maintains its proportion
combined_labels = data[diseases].apply(lambda x: ''.join(map(str, x)), axis=1)

# Add `combined_labels` as a new column to the `data`
data['combined_labels'] = combined_labels

# Split dataset by subject_id
subject_ids = data['subject_id'].unique()

# Generate a label (disease combination) for each subject_id
subject_labels = data.groupby('subject_id')['combined_labels'].first()

# Define a function for stratified splitting based on each label's proportions
def stratified_split(data, label_col, test_size=0.5):
    # Stratify each label's positive and negative samples
    train_data = pd.DataFrame()
    val_data = pd.DataFrame()

    for label in data[label_col].unique():
        label_data = data[data[label_col] == label]
        
        # Split the data for each label by the given proportion
        label_train, label_val = train_test_split(label_data, test_size=test_size, random_state=0)
        
        # Add the split train and validation data to the final datasets
        train_data = pd.concat([train_data, label_train], axis=0)
        val_data = pd.concat([val_data, label_val], axis=0)

    return train_data, val_data

# Use the custom stratified split function
train_data, val_data = stratified_split(data, label_col='combined_labels')

# Check the positive/negative sample ratios for each label in the train and validation sets
for disease in diseases:
    print(f"Train {disease} negative/positive: {train_data[disease].value_counts(normalize=True)}")
    print(f"Val {disease} negative/positive: {val_data[disease].value_counts(normalize=True)}")

# Create directories
# os.makedirs(train_folder, exist_ok=True)
# os.makedirs(val_folder, exist_ok=True)

# # Save the data to the respective folders
# train_data.to_csv(os.path.join( 'with_nonan_label_PA_train.csv'), index=False)
# val_data.to_csv(os.path.join('with_nonan_label_PA_val.csv'), index=False)

train_data.to_csv(os.path.join(val_folder, 'with_nonan_label_PA_val.csv'), index=False)
val_data.to_csv(os.path.join(test_folder, 'with_nonan_label_PA_test.csv'), index=False)
# test_data.to_csv(os.path.join(test_folder, 'with_nonan_label_PA_test.csv'), index=False)

print("Data has been split and saved successfully.")
