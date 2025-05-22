import pandas as pd

# Read CSV
df = pd.read_csv("data/final/dataset.csv")

# Write to Parquet using pyarrow for highest speed
df.to_parquet("data/final/dataset.parquet", engine="pyarrow", index=False, compression='snappy')
