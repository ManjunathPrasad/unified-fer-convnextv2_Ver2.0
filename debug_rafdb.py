import pandas as pd

# Read the individual RAF-DB prepared CSV
rafdb = pd.read_csv('emotion_q1_framework/Dataset/prepared/rafdb_prepared.csv')
print(f"RAF-DB prepared total: {len(rafdb)}")
print(f"RAF-DB label distribution:\n{rafdb['label'].value_counts().sort_index()}")
print()

# Read the unified CSV
unified = pd.read_csv('emotion_q1_framework/Dataset/prepared/unified_dataset.csv')
rafdb_in_unified = unified[unified['dataset'] == 'RAFDB']
print(f"RAF-DB in unified: {len(rafdb_in_unified)}")
print(f"Dropped: {len(rafdb) - len(rafdb_in_unified)} images")
print()

# Check if there are any NaN labels in RAF-DB prepared
print(f"RAF-DB NaN labels: {rafdb['label'].isna().sum()}")
print(f"RAF-DB non-numeric labels: {(~pd.to_numeric(rafdb['label'], errors='coerce').notna()).sum()}")
