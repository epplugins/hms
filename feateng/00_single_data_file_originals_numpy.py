# Saving all spectrograms as a single numpy file.

# Output: ../data/00_single_spectrograms_originals_numpy.npy
# Output: ../data/00_single_spectrograms_originals_numpy_items.npy

import os 
import pathlib
import numpy as np
import pandas as pd

base_dir = pathlib.Path("../../data/hms")

TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']

df_traincsv = pd.read_csv(f'{base_dir}/train.csv')
df_traincsv.loc[df_traincsv.expert_consensus == 'Seizure', 'target'] = 0
df_traincsv.loc[df_traincsv.expert_consensus == 'LPD', 'target'] = 1
df_traincsv.loc[df_traincsv.expert_consensus == 'GPD', 'target'] = 2
df_traincsv.loc[df_traincsv.expert_consensus == 'LRDA', 'target'] = 3
df_traincsv.loc[df_traincsv.expert_consensus == 'GRDA', 'target'] = 4
df_traincsv.loc[df_traincsv.expert_consensus == 'Other', 'target'] = 5

print("Loaded train.csv. Added target column.")

# For definitive code, change:
# - Slice of data, to include all files.
# - Increase divisor in remainder check.
# - Name of output files.

train_files_specs = os.listdir(f'{base_dir}/train_spectrograms')

# Saving columns of interest
specfile = train_files_specs[0]
spec_id = int(specfile.split('.')[0])
df_singlespec = pd.read_parquet(f'{base_dir}/train_spectrograms/{spec_id}.parquet')
columns = df_singlespec.columns[1:]

all = np.array([]).reshape(0,400)
sub_spectrograms_idxs = np.array([], dtype=int).reshape(0,4)
current = 0

for n, specfile in enumerate(train_files_specs):
    if n%200 == 0:
        print(f'{n} files loaded.')
    spec_id = int(specfile.split('.')[0])
    df_singlespec = pd.read_parquet(f'{base_dir}/train_spectrograms/{spec_id}.parquet')
    df_singlespec.replace(np.nan, 0, inplace=True)
    length = df_singlespec.shape[0]

    subs = df_traincsv.loc[df_traincsv.spectrogram_id == spec_id]
    for i in np.arange(len(subs)):
        item = subs.iloc[i]
        eeg_id = item.eeg_id
        eeg_sub_id = item.eeg_sub_id
        idx = current + int(item.spectrogram_label_offset_seconds/2)
        target = int(item.target)
        x = np.array([eeg_id, eeg_sub_id, idx, target]).reshape(1,4)
        sub_spectrograms_idxs = np.concatenate([sub_spectrograms_idxs, x])        
    x = df_singlespec[columns].to_numpy(copy=True)
    all = np.concatenate([all, x])
    current = current + x.shape[0]


np.save("../data/00_single_spectrograms_originals_numpy_items.npy", sub_spectrograms_idxs)
np.save("../data/00_single_spectrograms_originals_numpy.npy", all)

print('Saved to ../data/00_single_spectrograms_originals_numpy_items.npy')
print('Saved to ../data/00_single_spectrograms_originals_numpy.npy')


