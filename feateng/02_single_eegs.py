import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd

base_dir = "../../kaggle_data/hms"
# base_dir = "/kaggle/input/hms-harmful-brain-activity-classification"

df_traincsv = pd.read_csv(f'{base_dir}/train.csv')

#
# Cleaning.
#

eegs_with_nans = np.load("../EDA/02_eegs_with_nan_ids.npy")

# Counting NaN in each eeg.
eegs_with_nans_qty = np.array([], dtype=int)
for eeg_id in eegs_with_nans:
    eeg = pd.read_parquet(f'{base_dir}/train_eegs/{eeg_id}.parquet')
    qty = len(eeg[pd.isna(eeg).any(axis=1)].index)
    eegs_with_nans_qty = np.append(eegs_with_nans_qty, qty)

df_qtys = pd.DataFrame({'eeg_id': eegs_with_nans, 'quantity': eegs_with_nans_qty})

# Discarding eeg_ids with more than 5 NaNs.
eeg_ids_gt5 = df_qtys.loc[df_qtys.quantity > 5].eeg_id
idxs_to_remove = df_traincsv.loc[df_traincsv.eeg_id.isin(eeg_ids_gt5.values)].index
idxs = df_traincsv.index.difference(idxs_to_remove)
eeg_ids_clean = df_traincsv.iloc[idxs].eeg_id.unique()
print(f'{len(eeg_ids_clean)} eegs to load.')

#
# Populate dataframe.
#

df = pd.DataFrame()
for i, eeg_id in enumerate(eeg_ids_clean[0:6000]):
    if i%500 == 0:
        print(f'{i} files loaded.')
    eeg = pd.read_parquet(f'{base_dir}/train_eegs/{eeg_id}.parquet')
    eeg = eeg.interpolate(limit_direction='both') # <<<<< Interpolation

    data = np.ones(eeg.shape[0], dtype=np.int16)*eeg_id
    eeg.insert(0, 'eeg_id', data)
    df = pd.concat([df, eeg], ignore_index=True)

print('Saving to 02_single_eegs_00000_05999.parquet')
df.to_parquet('../data/02_single_eegs_00000_05999.parquet')