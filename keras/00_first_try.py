# Just testing how everything works.

import pandas as pd
import numpy as np
import keras
import pathlib
import os 

np.random.seed(536)

base_dir = pathlib.Path("/kaggle/input/hms-harmful-brain-activity-classification")
path_to_data = '/kaggle/input/hms-spectrograms-in-a-single-dataframe/00_single_spectrograms_originals.parquet'

df_traincsv = pd.read_csv(f'{base_dir}/train.csv')
df_traincsv.loc[df_traincsv.expert_consensus == 'Seizure', 'target'] = 0
df_traincsv.loc[df_traincsv.expert_consensus == 'LPD', 'target'] = 1
df_traincsv.loc[df_traincsv.expert_consensus == 'GPD', 'target'] = 2
df_traincsv.loc[df_traincsv.expert_consensus == 'LRDA', 'target'] = 3
df_traincsv.loc[df_traincsv.expert_consensus == 'GRDA', 'target'] = 4
df_traincsv.loc[df_traincsv.expert_consensus == 'Other', 'target'] = 5

TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']

print("Loaded train.csv. Added target column.")

# Tran/Val
ptrain = 0.8
n_total_samples = df_traincsv.shape[0]
cut = int(ptrain*n_total_samples)
idx = np.random.permutation(n_total_samples)
idx_train = idx[0:cut]
idx_val = idx[cut:]

#
# Data generator
#

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, items, path_to_data, batch_size=32, dim=(300,400), n_channels=1,
                 n_classes=6, shuffle=True):
        'Initialization'
        sel = [("spectrogram_id", "in", items['spectrogram_id'])]
        self.data = pd.read_parquet(path_to_data, filters=sel)
        self.data.replace(np.nan, 0, inplace=True)
        self.columns = self.data.columns[2:]
        self.dim = dim
        self.batch_size = batch_size
        # self.labels = labels
        self.items = items
        self.len = items.shape[0]
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.len / self.batch_size))+1

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.len)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        true_size = len(indexes)
        X = np.empty((true_size, *self.dim, self.n_channels))
        y = np.empty((true_size), dtype=int)

        # Generate data
        for i, idx in enumerate(indexes):
            item = self.items.iloc[idx]
            # Store sample
            X[i,] = self.data.loc[(self.data.spectrogram_id == int(item.spectrogram_id))&
               (self.data.time >= item.spectrogram_label_offset_seconds)&
               (self.data.time < item.spectrogram_label_offset_seconds + 600)][self.columns].to_numpy(copy=True).reshape((*self.dim,1))

            # Store class
            y[i] = int(item.target)

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


def make_model(input_shape, num_classes):
    input_layer = keras.layers.Input(input_shape)

    #max1 = keras.layers.MaxPooling1D(pool_size=2)(input_layer)
    
    conv1 = keras.layers.Conv2D(filters=32, kernel_size=3, padding="same")(input_layer)
    #conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.MaxPooling2D(pool_size=8)(conv1)
    conv1 = keras.layers.ReLU()(conv1)
    
    conv2 = keras.layers.Conv2D(filters=64, kernel_size=7, padding="same")(conv1)
    #conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.MaxPooling2D(pool_size=8)(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    # conv3 = keras.layers.Conv2D(filters=256, kernel_size=7, padding="same")(conv2)
    # #conv3 = keras.layers.BatchNormalization()(conv3)
    # conv3 = keras.layers.MaxPooling2D(pool_size=2)(conv3)
    # conv3 = keras.layers.ReLU()(conv3)

    # conv4 = keras.layers.Conv1D(filters=512, kernel_size=3, padding="same")(conv3)
    # conv4 = keras.layers.BatchNormalization()(conv4)
    # conv4 = keras.layers.MaxPooling1D(pool_size=4)(conv4)
    # conv4 = keras.layers.ReLU()(conv4)

    fltn  = keras.layers.Flatten()(conv2) 
    
    relu1 = keras.layers.Dense(256)(fltn)
    relu1 = keras.layers.ReLU()(relu1)

    relu2 = keras.layers.Dense(64)(relu1)
    relu2 = keras.layers.ReLU(64)(relu2)

    lin = keras.layers.Dense(2)(relu2)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(relu1)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


items_train = df_traincsv[['spectrogram_id','spectrogram_label_offset_seconds',
                      'target']].iloc[idx_train].reset_index(drop=True)
items_val = df_traincsv[['spectrogram_id','spectrogram_label_offset_seconds',
                      'target']].iloc[idx_val].reset_index(drop=True)

# Parameters
params = {
    'dim': (300,400),
    'batch_size': 32,
    'n_classes': 6,
    'n_channels': 1,
    'shuffle': True
    }

training_generator = DataGenerator(items_train, path_to_data, **params)
validation_generator = DataGenerator(items_val, path_to_data, **params)

model = make_model(input_shape=(*params['dim'],1), num_classes=params['n_classes'])

model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(training_generator, epochs=2, validation_data=validation_generator)

#
# Test Data generator: for predicting.
#

class test_DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, ids, path_to_test_data, batch_size=32, dim=(300,400), n_channels=1,
                n_classes=6):
        'Initialization'
        self.path = path_to_test_data
        # self.files = os.listdir(path_to_test_data)
        self.ids = ids
        self.indexes = np.arange(len(self.ids))
        # self.columns = self.data.columns[2:]
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        # items_temp = self.items.iloc[indexes]

        # Generate data
        X = self.__data_generation(indexes)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        # self.indexes = np.arange(self.len)
        # if self.shuffle == True:
        #     np.random.shuffle(self.indexes)
        pass

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((len(indexes), *self.dim, self.n_channels))

        # Generate data
        for i, idx in enumerate(indexes):
            # item = self.items.iloc[idx]
            test_spectrogram = pd.read_parquet(f'{self.path}{self.ids[idx]}.parquet')
            test_spectrogram.replace(np.nan, 0, inplace=True)

            # Store sample
            X[i,] = test_spectrogram.iloc[:,1:].to_numpy(copy=True).reshape((*self.dim,1))

        return X


# Parameters
params = {
    'dim': (300,400),
    'batch_size': 32,
    'n_classes': 6,
    'n_channels': 1,
    }

path_to_test_data = f'{base_dir}/test_spectrograms/'
test = pd.read_csv(f'{base_dir}/test.csv')
ids = test['spectrogram_id']

test_generator = test_DataGenerator(ids, path_to_test_data, **params)

y_pred = model.predict(test_generator)

sub = pd.DataFrame({'eeg_id':test.eeg_id.values})
sub[TARGETS] = np.round(y_pred,3)
sub.to_csv('submission.csv',index=False)
