import pandas as pd

rafdb = pd.read_csv('emotion_q1_framework/Dataset/prepared/rafdb_prepared.csv')
print("RAF-DB prepared columns:", rafdb.columns.tolist())
print("RAF-DB dataset column unique values:", rafdb['dataset'].unique() if 'dataset' in rafdb.columns else 'NO DATASET COLUMN')
print("First few rows:")
print(rafdb.head())
