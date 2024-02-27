# Saving all spectrograms as a single parquet file.

# Output: ../data/00_single_spectrograms_originals.parquet

import os 
import pathlib
import numpy as np
import pandas as pd

# base_dir = pathlib.Path("../../kaggle_data/hms")
# base_dir = pathlib.Path("/kaggle/input/hms-harmful-brain-activity-classification")
base_dir = pathlib.Path("../../data/hms")

train_files_specs = os.listdir(base_dir / "train_spectrograms")

df = pd.DataFrame()
for i, specfile in enumerate(train_files_specs):
    if i%100 == 0:
        print(f'Loaded {i} files.')
    spec_id = int(specfile.split('.')[0])
    df_singlespec = pd.read_parquet(f'{base_dir}/train_spectrograms/{spec_id}.parquet')
    data = np.ones(df_singlespec.shape[0], dtype=np.int16)*spec_id
    df_singlespec.insert(0, 'spectrogram_id', data)
    df = pd.concat([df,df_singlespec], ignore_index=True)

print('Saving to data/00_single_spectrograms_originals.parquet')
df.to_parquet('../data/00_single_spectrograms_originals.parquet')