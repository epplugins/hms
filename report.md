## EDA

### Given spectrograms: 

add detail.

Not useful.

### EEGs

**9999.0**
There are EEGs with lots of rows with values 9999.0. This looks like an open circuit or malfunction of some kind.

Only the 10 second central samples of each sub EEG are reliable, there no 9999.0 values in these.

Test set (in Kaggle): Some 10 s sub eegs have 9999.0 cells, but less than 0.5 s. (Further probing needs to be conducted to narrow down this number, perhaps it's just the initial values in the 10 s sample.)

**NaN** 

Only 244 sub eegs (50 s) have more than 2 NaN rows: discard these.  
There are 10 second sub EEG with rows where all cells are NaN.   
Only 46 have more than 1 NaN row.  
Only 39 have more than 2 NaN rows. Only one with 18 NaN rows and then 38 with more than 150 rows.  
Some have up to 1200 NaN rows (more than half the sample).

High number of NaN rows: discard. There are no NaN in the test set. See LB probing.

**Number of eeg_sub_id**

The longest running eegs (the ones with lots of sub eegs) are GPD, GRDA and LRDA. Only one or two of the other classes. (see 05_eegs_correlations)

**Changes in votes in sub eegs**

After discarding more than 2 NaN rows.  
1549 eegs have changes in votes percentages.

## Train/Val/Test

keras/03_stratified.ipynb

## Pipelines

Keras:  
Load indices for training, validation and testing sets. (03_stratified....)

Time for preprocessing 292 eegs, cwt: 225 s.  
There are between 2000 and 3500 eegs in LB test set, expect around 2300 s for preprocessing in LB.  


### Key points
- Discard sub eegs with more than 2 NaN rows. (indices: data/02_eegs_sub_50_idxs_less_3_nan.npy)
- Using observations where all sub eegs have the same voting probs. Needs further work for discrepancies.

## Runs

### Keras 10.ipynb



Spectrograms using only F3 and F4 in two channels.