data/00_single_spectrograms_originals.parquet

- pandas df with all full spectrograms, with spectrogram_id added.
- There are NaN values.

1. Load data/00_single_spectrograms_originals.parquet
1. Replace Nan with zero. NOT A GOOD IDEA. See fixed items.


data/02_single_eegs.parquet : not complete, too big?

- Pandas with all eegs from train.csv, after cleaning.

1. Removed eegs with more than 5 rows with NaN.
1. Interpolated NaN for the rest of eegs with NaN.