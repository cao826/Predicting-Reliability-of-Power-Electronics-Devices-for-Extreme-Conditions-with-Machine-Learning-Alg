import os
import sys
import warnings
import numpy as np
import tensorflow.keras as keras
import pickle
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.layers import LeakyReLU

def check_path(path):
    """ Checks if the pathh specified by the user is a path
    Arguments:
        path: path specified by user
    Returns:
        None
    """
    if (not os.path.isfile(path)):
        raise Exception('The path specified does not point to a file')

def check_if_correct_table(dataframe):
    """ Checks if the resulting dataframe read in from the path is the right data for encoder-decoder
    Arguments:
        dataframe: parameter type
    Returns:
        None
    """
    if (not dataframe.columns[0] == 'Unnamed: 0'):
        raise Exception("The dataframe you read is not the correct one for this script. Please double check that you inputted the path to the dataset for deep learning, and not traditional learning.")



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

"""
A quick function to reshape the data to form we need for encoder-decoder
"""
def prepare_for_encoder_decoder(data):
    """Take a dataframe and return the proper data formats"""
    sequence_data = data.iloc[:,1:2001].values.reshape((-1, 500, 4), order='F')
    target_data = data.iloc[:,-50:].values.reshape((-1, 50, 1))
    shift_blank = np.zeros(target_data.shape)
    shift_blank[:,1:, :] = target_data[:,:-1, :]
    
    encoder_input_data = sequence_data
    decoder_input_data = target_data
    decoder_target_data = shift_blank
    return encoder_input_data, decoder_input_data, decoder_target_data

def encoder_decoder_v1():
    """
    Returns the different models needed to train and
    test an encoder-decoder model in python
    Returns:
        model - used to link the encoder_model and decoder_model. This is also what we train
        encoder_model
        decoder_model
    """
    latent_dim = 128 #this is the number of cells ("neurons") we want our layer to have

    #training_encoder
    encoder_input = Input(shape=(500,4))
    encoder = LSTM(latent_dim, return_state=True) #recover the enoder states, which we pass to the decoder
    encoder_outputs, state_h, state_c = encoder(encoder_input)
    encoder_states = [state_h, state_c]

    #training decoder
    decoder_inputs = Input(shape=(None, 1)) #1 is the number of features per timestep in the output sequence
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)

    decoder_outputs, u1, y2 = decoder_lstm(decoder_inputs,
                                           initial_state=encoder_states)
    decoder_dense = Dense(1, activation=keras.activations.sigmoid)
    decoder_lrelu = LeakyReLU()
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_input, decoder_inputs], decoder_outputs)

    #inference encoder
    encoder_model = Model(encoder_input, encoder_states)

    #inference decoder
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model

def predict_sequence(infenc, infdec, source, n_steps, cardinality):
    # encode
    state = infenc.predict(source)
    # start of sequence input
    target_seq = np.array([0.0 for _ in range(cardinality)]).reshape(1, 1, cardinality)
    # collect predictions
    output = list()
    for t in range(n_steps):
        # predict next char
        yhat, h, c = infdec.predict([target_seq] + state)
        # store prediction
        output.append(yhat[0,0,:])
        # update state
        state = [h, c]
        # update target sequence
        target_seq = yhat
    return np.array(output)
