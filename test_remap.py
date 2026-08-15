import pandas as pd
from label_semantics import map_labels_to_emotions

# Read the RAF-DB prepared CSV
rafdb = pd.read_csv('emotion_q1_framework/Dataset/prepared/rafdb_prepared.csv')
print(f"Before remap: {len(rafdb)} rows, labels: {sorted(rafdb['label'].unique())}")

# Apply the remap
rafdb['dataset'] = 'RAFDB'
rafdb_remapped = map_labels_to_emotions(rafdb, 'RAFDB')
print(f"After remap: {len(rafdb_remapped)} rows, labels: {sorted(rafdb_remapped['label'].unique())}")

# Show what changed
dropped = len(rafdb) - len(rafdb_remapped)
print(f"Dropped: {dropped} rows")
