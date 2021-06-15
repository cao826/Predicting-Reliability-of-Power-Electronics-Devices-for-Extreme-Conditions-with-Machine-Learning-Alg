import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit
import tensorflow.keras as keras
import tensorflow.keras.optimizers as optimizers # pylint: disable=import-error
from tensorflow.keras.layers import (Input, # pylint: disable=import-error
                                     Concatenate,
                                     LSTM,
                                     Bidirectional,
                                     Flatten,
                                     Dense,
                                     MaxPooling2D,
                                     Dropout,
                                     Concatenate,
                                     LeakyReLU)
from tensorflow.keras.models import Model, Sequential # pylint: disable=import-error
from sklearn.ensemble import GradientBoostingRegressor

lrelu = LeakyReLU()

def device_ids_with_k(k, device_ids):
    """ From a list of device ids, returns a testing set of all k type devices (A_1, B_1, etc.)
    Arguments:
        k: int
        device_ids: list
    Returns:
        List denoting testing and training device ids
    """
    k_string = '_{}'.format(k)
    k_devices = [device for device in device_ids if device.endswith(k_string)]
    return k_devices

def get_fold_k(k, data):
    """ Partitions data into a training set of all k devices (A_1, B_2, etc.) and a testing set.
    Arguments:
        k: int
        data: pandas.DataFrame object
    Returns:
        Two pandas.DataFrame objects, denoting the testing and training set
    """
    devices = list(data['Device ID'].values)
    testing_devices = device_ids_with_k(k, devices)
    testing_set = data[data['Device ID'].isin(testing_devices)]
    training_set = data[~data['Device ID'].isin(testing_devices)]
    return testing_set, training_set

def multimodal_lstm_v1():
    optimizer = optimizers.Adam(lr=.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    loss = 'mean_squared_error'
    metrics = ['mean_absolute_error']

    sequence_input = Input(shape=(500,4))
    lstm = LSTM(50, input_shape=(500, 4))(sequence_input)
    nonsequence_input= Input(12)
    combined = Concatenate()([lstm, nonsequence_input])
    #print(combined)
    combined_hidden = Dense(55, activation=lrelu)(combined)
    out = Dense(50)(combined_hidden)
    model = Model(inputs=[sequence_input, nonsequence_input], outputs=out)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    return model

def multimodal_lstm_vsig():
    optimizer = optimizers.Adam(lr=.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    loss = 'mean_squared_error'
    metrics = ['mean_absolute_error']
    sig = keras.activations.sigmoid

    sequence_input = Input(shape=(500,4))
    lstm = LSTM(50, input_shape=(500, 4))(sequence_input)
    nonsequence_input= Input(12)
    combined = Concatenate()([lstm, nonsequence_input])
    combined_hidden = Dense(55, activation=sig)(combined)
    out = Dense(50)(combined_hidden)
    model = Model(inputs=[sequence_input, nonsequence_input], outputs=out)
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)
    return model
