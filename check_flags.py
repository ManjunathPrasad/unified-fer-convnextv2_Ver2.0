import pandas as pd

rafdb = pd.read_csv('emotion_q1_framework/Dataset/prepared/rafdb_prepared.csv')
print("Columns:", rafdb.columns.tolist())
print("Has label_mapped?", 'label_mapped' in rafdb.columns)
print("Has label_name?", 'label_name' in rafdb.columns)

if 'label_name' in rafdb.columns:
    print("label_name values:", rafdb['label_name'].unique()[:5])
