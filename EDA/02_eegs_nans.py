#
# Find missing values in eegs.
#

import numpy as np
import pandas as pd

# base_dir = "../../kaggle_data/hms"
base_dir = "../../data/hms"
# base_dir = "/kaggle/input/hms-harmful-brain-activity-classification"

df_traincsv = pd.read_csv(f'{base_dir}/train.csv')
eeg_ids = df_traincsv.eeg_id.unique()

eegs_with_nans = np.array([], dtype=int)
for i, eeg_id in enumerate(eeg_ids):
    if i%500 == 0:
        print(f'{i} eegs tested.')

    eeg = pd.read_parquet(f'{base_dir}/train_eegs/{eeg_id}.parquet')
    if pd.isna(eeg).any(axis=None):
        eegs_with_nans = np.append(eegs_with_nans, eeg_id)

print(f'Number of eegs with NaN: {len(eegs_with_nans)}')

np.save("02_eegs_with_nan_ids.npy", eegs_with_nans)
