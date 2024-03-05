# Participation in HMS - Harmful Brain Activity Classification

Classify seizures and other patterns of harmful brain activity in critically ill patients

https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification

## TODO

- LB probing: Count number of NaN rows in 10 seconds samples. Submit different predictions for observations with high number of NaN rows, to probe if there are some.

- Train/Val split: 
    - Reduce overrepresentation: target repeated many times in the same eeg. Perhaps reduce the number of observations to get a balanced dataset.
    - Unanimous consensus.
- Create new targets when there consensus is not unanimous: complete clustering (00_looking_around.ipynb)
- Reduce number of observations, keeping only observations with high number of votes.
- Outliers: in spectrograms, intensities?
- class weight
- feature extraction
- Determine if Relu or leakyrelu: if I do normalization around 0, I could use leaky (because of negative values)
- Fine tunning CNN: Relu parameters, L2 value, number of epochs
- Download dataset again in lenovo.
- Single file pandas: needs to be redone if necessary.
- there are lots of NA in spectrograms, I guess that's correct, but they should be converted to 0?
- Normalization, yes or no? Batchnormalization layer is needed in CNN.
- Rounding predictions. I'm rounding at 3 decimals, how many decimals is ok?
- Reshaping numpy: check this step, maybe some tests with other images.
- banana montage: banda pass before or after subtraction?

## Pipeline

1. Load idxs for train/validation. (04_eda_patient.ipynb) There are three options:
    - Without removing repeated observations: use code in "v1".
    - Reducing possible oversampling.
    - Unanimous consensus.

## Search:
- Artifacts in eegs, there are models.

## Feature Engineering

1. EEGs with NaNs.
02_eegs_idxs_up_to_5_nan.csv has the indices of train.csv with eegs with up to 5 NaN.

TODO:
- find a way to avoid interpolating NaNs each time. Perhaps saving a new dataset with the interpolated files.

denoising
band pass (alpha, beta, theta bands)

Feature extraction methods include the Fast Fourier Transform, Discrete
Cosine Transform, Poincare, Power Spectral Density, Hjorth parameters, and some statistical features. The Chi-
square and Recursive Feature Elimination procedures were used to choose the discriminative features among
them.

he great majority of
ocular artifacts occur below 4 Hz, muscular motions occur above 30 Hz, and power line noise occurs between
50 and 60 Hz 3



## Links

Multiclass classification, imbalance, some EDA:

https://www.aitude.com/multiclass-classification-on-highly-imbalanced-dataset/

Data generator for keras:

https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

https://github.com/shervinea/enzynet

Filtering in python:

https://swharden.com/blog/2020-09-23-signal-filtering-in-python/

https://www.learningeeg.com/epileptiform-activity  


wavelets:  
https://ataspinar.com/2018/12/21/a-guide-for-using-the-wavelet-transform-in-machine-learning/  
https://pywavelets.readthedocs.io/en/latest/  




## Environments

1. gsdc

    Versions on Kaggle.

1. latest

    Updated

1. pyemd
    Since there is no conda package for ceemd, it's best to use a separate environment for this package, that needs to be installed using pip.

    ``conda create -n pyemd -c conda-forge python=3.10``  
    ``conda install -n pyemd -c conda-forge numpy pandas scipy matplotlib ipykernel``  
    ``conda activate pyemd``  
    ``pip install EMD-signal``  


## References

CEEMD and XGBoost:  
Wu, J.; Zhou, T.; Li, T. Detecting Epileptic Seizures in EEG Signals with Complementary Ensemble Empirical Mode Decomposition and Extreme Gradient Boosting. Entropy 2020, 22, 140. https://doi.org/10.3390/e22020140


pyemd  
https://pyemd.readthedocs.io/en/latest/intro.html  
``@misc{pyemd,
  author = {Laszuk, Dawid},
  title = {Python implementation of Empirical Mode Decomposition algorithm},
  year = {2017},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/laszukdawid/PyEMD}},
  doi = {10.5281/zenodo.5459184}
}``

pyWavelets  
Gregory R. Lee, Ralf Gommers, Filip Wasilewski, Kai Wohlfahrt, Aaron O’Leary (2019). PyWavelets: A Python package for wavelet analysis. Journal of Open Source Software, 4(36), 1237, https://doi.org/10.21105/joss.01237.  




## Download

