import sys
import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import (ModelCheckpoint,
                             TensorBoard, ReduceLROnPlateau,
                             CSVLogger, EarlyStopping)
from keras.backend.tensorflow_backend import set_session
from model import model
import argparse
from keras.utils import HDF5Matrix
import pandas as pd
import h5py
import numpy as np

if __name__ == "__main__":
    # Get data and train
    parser = argparse.ArgumentParser(description='Train neural network.')
    parser.add_argument('path_to_hdf5', type=str,
                        help='path to hdf5 file containing tracings')
    parser.add_argument('path_to_csv', type=str,
                        help='path to csv file containing annotations')
    parser.add_argument('--val_split', type=float, default=0.02,
                        help='number between 0 and 1 determining how much of'
                             ' the data is to be used for validation. The remaining '
                             'is used for validation. Default: 0.02')
    parser.add_argument('--dataset_name', type=str, default='tracings',
                        help='name of the hdf5 dataset containing tracings')
    args = parser.parse_args()
    # Optimization settings
    loss = 'binary_crossentropy'
    lr = 0.001
    batch_size = 64
    opt = Adam(lr)
    callbacks = [ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=7,
                                   min_lr=lr / 100),
                 EarlyStopping(patience=9,  # Patience should be larger than the one in ReduceLROnPlateau
                               min_delta=0.00001)]
    # Set session and compile model
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    # If you are continuing an interrupted section, uncomment line bellow:
    #   model = keras.models.load_model(PATH_TO_PREV_MODEL, compile=False)
    model.compile(loss=loss, optimizer=opt)
    # Get annotations
    y = pd.read_csv(args.path_to_csv).values
    # Get tracings
    f = h5py.File(args.path_to_hdf5, "r")
    x = f[args.dataset_name]

    # Create log
    callbacks += [TensorBoard(log_dir='./logs', batch_size=batch_size, write_graph=False),
                  CSVLogger('training.log', append=False)]  # Change append to true if continuing training
    # Save the BEST and LAST model
    callbacks += [ModelCheckpoint('./backup_model_last.hdf5'),
                  ModelCheckpoint('./backup_model_best.hdf5', save_best_only=True)]
    # Train neural network
    history = model.fit(x, y,
                        batch_size=batch_size,
                        epochs=70,
                        initial_epoch=0,  # If you are continuing a interrupted section change here
                        validation_split=args.val_split,
                        shuffle='batch',  # Because our dataset is an HDF5 file
                        callbacks=callbacks,
                        verbose=1)
    # Save final result
    model.save("./final_model.hdf5")
    f.close()
