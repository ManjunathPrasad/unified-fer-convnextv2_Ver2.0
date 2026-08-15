# Quick test: does the remap actually work on the raw RAFDB data?
import pandas as pd
from label_semantics import RAFDB_RAW_TO_CANONICAL

# Simulate raw RAF-DB labels
labels = pd.Series([1, 2, 3, 4, 5, 6, 7])
print("Original labels:", labels.tolist())

# Apply the map
mapped = labels.map(RAFDB_RAW_TO_CANONICAL)
print("After map:", mapped.tolist())
print("Expected: [5, 2, 1, 3, 4, 0, 6]")
print("Missing values?", mapped.isna().sum())
