"""Run the EnzyNet architecture."""

# Authors: Afshine Amidi <lastname@mit.edu>
#          Shervine Amidi <firstname@stanford.edu>

# MIT License

from absl import app
from absl import flags

import os

from enzynet import constants
from enzynet import keras_utils
from enzynet import models
from enzynet import tools
from enzynet import volume
from keras import backend as K
from keras import callbacks
from keras import optimizers

FLAGS = flags.FLAGS

# Main parameters.
flags.DEFINE_enum('mode_dataset', default='full',
                  enum_values=['full', 'reduced'],
                  help='Version of the dataset to use. "full" represents the '
                  'entire dataset whereas "reduced" denotes just a 10% fixed '
                  'random subset.')
flags.DEFINE_enum('mode_run', default='test', enum_values=['train', 'test'],
                  help='Whether to run the training or testing portion of the '
                  'code.')

# Volume parameters.
flags.DEFINE_list('weights', default=[], help='Weights to be used along with '
                  'coordinates to characterize PDB volumes. Each weight type '
                  'specified here adds an additional channel.')
flags.DEFINE_bool('scaling_weights', default=True, help='If set to True, '
                  'weight values are scaled to have a maximum magnitude of 1. '
                  'Note: this parameter is ignored if no weights are '
                  'specified.')
flags.DEFINE_integer('p', lower_bound=0, default=0, help='Number of '
                     'interpolated points between two consecutive represented '
                     'atoms. This parameter is used for finer grid '
                     'representations in order to draw lines between '
                     'consecutive points.')
flags.DEFINE_integer('v_size', lower_bound=1, default=32, help='Size of each '
                     'dimension of the grid where enzymes are represented. The '
                     'paper used a value of 32 but higher values such as 64 or '
                     '96 (cf. Figure 2 of the paper) could prove to carry '
                     'finer, more useful structural information. In this case, '
                     'please note that the architecture would also likely need '
                     'an upgrade in order to see performance gains.')
flags.DEFINE_float('max_radius', lower_bound=0.1, default=40, help='Maximum '
                   'enzyme radius (in angstroms) that entirely fits in the '
                   'volume. A higher value means more information is included '
                   'in the grid at a coarser resolution. Dataset statistics '
                   'are provided in Figure 4 of the paper.')
flags.DEFINE_bool('shuffle', default=True, help='Whether to shuffle the order '
                  'of dataset exploration in-between training epochs.')
flags.DEFINE_bool('noise_treatment', default=False, help='Whether to remove '
                  'isolated atoms from the enzyme volume. This is makes the '
                  'structure visually more coherent at fine grid sizes (> 64) '
                  'but is an almost no-op at lower resoltions (e.g. 32).')
flags.DEFINE_float('flip_probability', lower_bound=0.0, upper_bound=1.0,
                   default=0.2, help='Probability of flipping the enzyme with '
                   'respect to any axis. Used as an augmentation technique '
                   'during training.')

# Training parameters.
flags.DEFINE_integer('batch_size', lower_bound=1, default=32, help='Training '
                     'and validation batch size.')
flags.DEFINE_integer('max_epochs', lower_bound=1, default=200, help='Number of '
                     'training epochs.')
flags.DEFINE_enum('mode_weights', default='unbalanced',
                  enum_values=['unbalanced', 'balanced'],
                  help='Characterization of the strategy for mistake '
                  'penalization in the calculation of the loss.')
flags.DEFINE_integer('period_checkpoint', lower_bound=1, default=50,
                     help='Epoch frequency at which model weights are saved '
                     'during the training process.')

# Testing parameters.
flags.DEFINE_enum('voting_type', default='probabilities',
                  enum_values=['classes', 'probabilities'],
                  help='Determines how voting decisions are merged. '
                  '"probabilities" is used when the predicted class is the '
                  'argmax of the sum of probabilities, "classes" when the '
                  'argmax operates on the sum of predicted classes.')
flags.DEFINE_list('augmentation', default=['None', 'flips', 'weighted_flips'],
                  help='Denotes the augmentation techniques used at testing '
                  'time.')

# Miscellaneous.
LOSS_TYPE_TO_NAME = {
    'unbalanced': 'enzynet_uniform',
    'balanced': 'enzynet_adapted',
}


def main(_):
    ##---------------------------- Dataset -----------------------------------##
    # Load dictionary of labels.
    DICTIONARY = tools.read_dict(
        os.path.join(constants.DATASETS_DIR, 'dataset_single.csv'),
        value_type=constants.ValueType.INT)

    # Load partitions.
    if FLAGS.mode_dataset == 'full':
        partition = tools.read_dict(
            os.path.join(constants.DATASETS_DIR, 'partition_single.csv'),
            value_type=constants.ValueType.LIST_STRING)
    elif FLAGS.mode_dataset == 'reduced':
        partition = tools.read_dict(
            os.path.join(constants.DATASETS_DIR, 'partition_single_red.csv'),
            value_type=constants.ValueType.LIST_STRING)

    # Final computations.
    partition['train'] = partition['train'] + partition['validation']
    partition['validation'] = partition['test']

    # Get class weights and run type.
    class_weights = tools.get_class_weights(DICTIONARY, partition['train'],
                                            mode=FLAGS.mode_weights)
    run_type = LOSS_TYPE_TO_NAME[FLAGS.mode_weights]

    # Training generator.
    training_generator = volume.VolumeDataGenerator(
        list_enzymes=partition['train'],
        labels=DICTIONARY,
        v_size=FLAGS.v_size,
        flips=(FLAGS.flip_probability,) * constants.N_DIMENSIONS,
        batch_size=FLAGS.batch_size,
        shuffle=FLAGS.shuffle,
        p=FLAGS.p,
        max_radius=FLAGS.max_radius,
        noise_treatment=FLAGS.noise_treatment,
        weights=FLAGS.weights,
        scaling_weights=FLAGS.scaling_weights)

    # Validation generator.
    validation_generator = volume.VolumeDataGenerator(
        list_enzymes=partition['validation'],
        labels=DICTIONARY,
        v_size=FLAGS.v_size,
        flips=(0,) * constants.N_DIMENSIONS,  # No flip.
        batch_size=FLAGS.batch_size,
        shuffle=False,  # Validate with fixed set.
        p=FLAGS.p,
        max_radius=FLAGS.max_radius,
        noise_treatment=FLAGS.noise_treatment,
        weights=FLAGS.weights,
        scaling_weights=FLAGS.scaling_weights)

    # Check if data has been precomputed.
    training_generator.check_precomputed()

    ##--------------------------- Testing ------------------------------------##
    # Voting object.
    predictions = keras_utils.Voting(
        list_enzymes=partition['test'],
        labels=DICTIONARY,
        voting_type=FLAGS.voting_type,
        v_size=FLAGS.v_size,
        augmentation=FLAGS.augmentation,
        p=FLAGS.p,
        max_radius=FLAGS.max_radius,
        noise_treatment=FLAGS.noise_treatment,
        weights=FLAGS.weights,
        scaling_weights=FLAGS.scaling_weights)

    ##---------------------------- Model -------------------------------------##
    # Retrieve EnzyNet model architecture.
    model = models.enzynet(FLAGS.v_size, n_channels=1+len(FLAGS.weights))

    # Track accuracy and loss in real-time.
    history = keras_utils.MetricsHistory(saving_path=run_type + '.csv')

    # Checkpoints.
    checkpoints = callbacks.ModelCheckpoint(
        os.path.join(
            constants.CHECKPOINTS_DIR, f'{run_type}_{{epoch:02d}}.hd5f'),
        save_weights_only=True,
        period=FLAGS.period_checkpoint)

    if FLAGS.mode_run == 'train':
        # Compile.
        model.compile(optimizer=optimizers.Adam(lr=0.001, decay=0.00016667),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # Train.
        model.fit_generator(generator=training_generator,
                            epochs=FLAGS.max_epochs,
                            verbose=1,
                            validation_data=validation_generator,
                            callbacks=[history, checkpoints],
                            class_weight=class_weights,
                            use_multiprocessing=True,
                            workers=8,
                            max_queue_size=30)

    if FLAGS.mode_run == 'test':
        # Load weights.
        weights_path = os.path.join(constants.CHECKPOINTS_DIR,
                                    f'{run_type}_{FLAGS.max_epochs:02d}.hd5f')
        model.load_weights(weights_path)

    # Predict.
    predictions.predict(model)

    # Compute indicators.
    predictions.get_assessment()

    # Clear session.
    K.clear_session()


if __name__ == '__main__':
    app.run(main)
