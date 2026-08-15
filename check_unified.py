import pandas as pd

# Check the unified CSV more carefully
unified = pd.read_csv('emotion_q1_framework/Dataset/prepared/unified_dataset.csv')

# Check RAF-DB rows specifically
rafdb_unified = unified[unified['dataset'] == 'RAFDB']
print(f"RAF-DB in unified: {len(rafdb_unified)}")
print(f"RAF-DB labels in unified: {sorted(rafdb_unified['label'].unique())}")
print(f"RAF-DB NaN labels: {rafdb_unified['label'].isna().sum()}")
print()

# Compare: what happened to the raw prepared CSV?
rafdb_prepared = pd.read_csv('emotion_q1_framework/Dataset/prepared/rafdb_prepared.csv')
print(f"RAF-DB in prepared: {len(rafdb_prepared)}")
print(f"RAF-DB labels in prepared: {sorted(rafdb_prepared['label'].unique())}")
print()

# Check if label 7 survived or got filtered
label_7_prepared = (rafdb_prepared['label'] == 7).sum()
label_6_unified = (rafdb_unified['label'] == 6).sum()
print(f"Label 7 in prepared: {label_7_prepared}")
print(f"Label 6 in unified: {label_6_unified}")
print(f"Match? {label_7_prepared == label_6_unified}")
