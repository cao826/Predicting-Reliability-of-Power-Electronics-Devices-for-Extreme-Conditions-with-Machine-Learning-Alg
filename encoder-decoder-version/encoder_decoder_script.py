import os
import argparse as arg
import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow.keras as keras
import pickle
import matplotlib.pyplot as plt
import encoder_decoder_module as enc
import tensorflow.keras as keras
import pickle
import matplotlib.pyplot as plt

parser = arg.ArgumentParser(description="Produces predicted points with LSTM based encoder-decoder models.")
parser.add_argument('Path',
        metavar='path',
        type=str,
        help='Path to the data for deep learning')

args = parser.parse_args()
path = args.Path
enc.check_path(path)
data = pd.read_csv(path)
enc.check_if_correct_table(data)

new_columns = data.columns.values
new_columns[0] = 'index'
data.columns = new_columns
data = data.set_index('index')

lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                             factor=0.1,
                                             patience=4,
                                             verbose=1,
                                             min_lr=.0000001)
callbacks = [lr_scheduler]

encoder_decoder_predictions = dict()
for i in range(1, 25):
    test_data, train_data = enc.get_fold_k(i, data)
    test_encoder_input, test_decoder_input, test_decoder_target = enc.prepare_for_encoder_decoder(test_data)
    train_encoder_input, train_decoder_input, train_decoder_target = enc.prepare_for_encoder_decoder(train_data)
    model, encoder_model, decoder_model = enc.encoder_decoder_v1()
    model.compile(optimizer='adam', loss='mean_absolute_error')
    model.fit([train_encoder_input, train_decoder_input], train_decoder_target,
              batch_size=20,
              epochs=90,
              validation_split=0.1,
             callbacks=callbacks)

    for i in range(test_encoder_input.shape[0]):
        test_input = test_encoder_input[i, :,:].reshape(1, 500, 4)
        pred_seq = list(enc.predict_sequence(encoder_model, decoder_model, test_input, 50, 1).reshape(50))
        device_id = test_data['Device ID'].iloc[i]
        encoder_decoder_predictions[device_id] = pred_seq
    break
with open('encoder_decoder_predictions.pickle', 'wb') as handle:
    pickle.dump(encoder_decoder_predictions, handle)
