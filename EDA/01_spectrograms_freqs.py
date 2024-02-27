# Count the appearences of each frequency for each lobe.

# Result: 11138 counts for all frequencies in each group.


import os 
import pathlib
import numpy as np
import pandas as pd

base_dir = pathlib.Path("../../data/hms")
train_files_specs = os.listdir(base_dir / "train_spectrograms")

frequencies = {
    'LL' : np.array([]),
    'LP' : np.array([]),
    'RL' : np.array([]),
    'RP' : np.array([])
}
for file in train_files_specs:
    df_spec = pd.read_parquet(f'{base_dir}/train_spectrograms/{file}')
    columns = df_spec.columns[1:]
    for col in columns:
        desc = col[0:2]
        freq = float(col[3:])
        frequencies[desc] = np.concatenate([frequencies[desc], [freq]])

for tt in ['LL', 'LP', 'RL', 'RP']:
    val, counts = np.unique(frequencies['LL'], return_counts=True)
    print(np.unique(counts))
