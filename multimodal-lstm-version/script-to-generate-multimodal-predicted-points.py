import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit
import tensorflow.keras as keras
import tensorflow.keras.optimizers as optimizers # pylint: disable=import-error
import multimodal_module as mult

path = "/Users/calchuchesta/Box/Prime technical folder ML and AI work/Carlos's Folder/Summer_2021/data/modeling-data-encoder-decoder-format.csv"

data = pd.read_csv(path, index_col=0)
print(data)

early_stop = keras.callbacks.EarlyStopping(monitor='loss',
                                       patience=8,
                                       verbose=1)
lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                             factor=0.1,
                                             patience=4,
                                             verbose=1,
                                             min_lr=.0000001)
callbacks = [early_stop, lr_scheduler]

predicted_curves = dict()
for k in range(1,25):
    ## get data ready for input to model
    fold_k_test, fold_k_train = mult.get_fold_k(k, data)
    X_train_sequence_data = fold_k_train.iloc[:,1:2001].values
    first_dim = X_train_sequence_data.shape[0]
    X_train_sequence_data_for_input = X_train_sequence_data.reshape((first_dim, 500, 4), order='F')
    X_train_nonsequence_data = fold_k_train.iloc[:,2001:2013].values
    y_train = fold_k_train.iloc[:,-50:].values

    ##Get testing_data ready
    X_test_sequence_data = fold_k_test.iloc[:,1:2001].values
    first_dim = X_test_sequence_data.shape[0]
    X_test_sequence_data_for_input = X_test_sequence_data.reshape((first_dim, 500, 4), order='F')
    X_test_nonsequence_data = fold_k_test.iloc[:,2001:2013].values


    #get model ready
    model = mult.multimodal_lstm_v1()
    model.fit([X_train_sequence_data_for_input, X_train_nonsequence_data], y_train, batch_size=20,
                                                                                    epochs=1, callbacks=callbacks)
    preds = model.predict([X_test_sequence_data_for_input, X_test_nonsequence_data])

    ## matching predicted curves to device_ids and putting them in a dict
    num_test_ids = preds.shape[0]
    for i in range(num_test_ids):
        print
        device_id = fold_k_test['Device ID'].iloc[i]
        predicted_curve = preds[i,:]
        predicted_curves[device_id] = predicted_curve
    break

with open('multimodal-preds.pickle', 'wb') as f:
    pickle.dump(predicted_curves, f)
