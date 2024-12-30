import pandas as pd
import os
import math

# Read the large CSV file
df = pd.read_csv('vector_store/embeddings.csv')

# Calculate number of chunks needed (each chunk should be ~20MB to be safe)
total_rows = len(df)
chunk_size = math.ceil(total_rows / 6)  # Split into 6 parts to ensure each is under 25MB

# Create chunks directory if it doesn't exist
os.makedirs('vector_store/embeddings_chunks', exist_ok=True)

# Split the dataframe and save chunks
for i, chunk_start in enumerate(range(0, total_rows, chunk_size)):
    chunk = df[chunk_start:chunk_start + chunk_size]
    chunk.to_csv(f'vector_store/embeddings_chunks/embeddings_part_{i+1}.csv', index=False)
    print(f'Created part {i+1} with {len(chunk)} rows')
