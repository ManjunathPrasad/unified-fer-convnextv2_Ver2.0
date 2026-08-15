import pandas as pd
from label_semantics import map_labels_to_emotions, is_already_mapped

# Simulate what unify() does
fer_df = pd.read_csv('emotion_q1_framework/Dataset/prepared/fer2013_prepared.csv')
raf_df = pd.read_csv('emotion_q1_framework/Dataset/prepared/rafdb_prepared.csv')
aff_df = pd.read_csv('emotion_q1_framework/Dataset/prepared/affectnet_prepared.csv')

all_df = pd.concat([fer_df, raf_df, aff_df], ignore_index=True)
print(f"Before remap: {len(all_df)} total, RAF-DB: {len(all_df[all_df['dataset']=='RAFDB'])}, labels: {sorted(all_df[all_df['dataset']=='RAFDB']['label'].unique())}")

# Check if the RAF-DB subset is detected as already mapped
raf_subset = all_df[all_df['dataset'] == 'RAFDB']
print(f"is_already_mapped(RAF-DB subset)? {is_already_mapped(raf_subset)}")

# Now call map_labels_to_emotions like unify() does
for dataset_name in ['FER2013', 'RAFDB', 'AffectNet']:
    before = len(all_df[all_df['dataset']==dataset_name])
    all_df = map_labels_to_emotions(all_df, dataset_name)
    after = len(all_df[all_df['dataset']==dataset_name])
    print(f"{dataset_name}: before={before}, after={after}, labels={sorted(all_df[all_df['dataset']==dataset_name]['label'].unique())}")

print(f"\nAfter remap: {len(all_df)} total, RAF-DB: {len(all_df[all_df['dataset']=='RAFDB'])}, labels: {sorted(all_df[all_df['dataset']=='RAFDB']['label'].unique())}")
