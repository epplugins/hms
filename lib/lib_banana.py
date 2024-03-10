
import numpy as np
import pandas as pd
from scipy.signal import sosfiltfilt, butter


# 2024022800
def banana(eeg_absolute, filter=False, fs=200.0):
    '''Returns pandas dataframe with a banana montage.

    filter: False or [low freq, high freq]
    '''
    if filter:
        filtered_data = eeg_absolute.copy()
        # Apply band pass.
        sos = butter(5, filter, btype='bandpass', fs=fs, output='sos')
        for c in filtered_data.columns:
            filtered_data[c] = sosfiltfilt(sos, filtered_data[c])
    else:
        filtered_data = eeg_absolute.copy()

    eeg = pd.DataFrame(data={
        'Fp1-F7' : filtered_data.Fp1 - filtered_data.F7,
        'Fp7-T3' : filtered_data.F7 - filtered_data.T3,
        'T3-T5' : filtered_data.T3 - filtered_data.T5,
        'T5-O1' : filtered_data.T5 - filtered_data.O1,

        'Fp2-F8' : filtered_data.Fp2 - filtered_data.F8,
        'F8-T4' : filtered_data.F8 - filtered_data.T4,
        'T4-T6' : filtered_data.T4 - filtered_data.T6,
        'T6-O2' : filtered_data.T6 - filtered_data.O2,

        'Fp1-F3' : filtered_data.Fp1 - filtered_data.F3,
        'F3-C3' : filtered_data.F3 - filtered_data.C3,
        'C3-P3' : filtered_data.C3 - filtered_data.P3,
        'P3-O1' : filtered_data.P3 - filtered_data.O1,

        'Fp2-F4' : filtered_data.Fp2 - filtered_data.F4,
        'F4-C4' : filtered_data.F4 - filtered_data.C4,
        'C4-P4' : filtered_data.C4 - filtered_data.P4,
        'P4-O2' : filtered_data.P4 - filtered_data.O2,

        'Fz-Cz' : filtered_data.Fz - filtered_data.Cz,
        'Cz-Pz' : filtered_data.Cz - filtered_data.Pz,

        'EKG' : filtered_data.EKG
        })
    return eeg
