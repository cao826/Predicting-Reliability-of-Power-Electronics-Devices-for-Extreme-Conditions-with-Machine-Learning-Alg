import pandas as pd
from tqdm import tqdm
import gradboost_module as gmod
import pickle

path = "/Users/calchuchesta/Box/Prime technical folder ML and AI work/Carlos's Folder/Summer_2021/data/modeling-data.csv"

data_w_labels = pd.read_csv(path)


predictions = dict()
for i in range(1, 25):
    test_data, train_data = gmod.get_fold_k(i, data_w_labels)
    X_train = train_data.iloc[:, 1:-51].values
    Y_train = train_data.iloc[:, -51:-1].values
    X_test = test_data.iloc[:, 1:-51].values
    preds_matrix = gmod.xgboostmult(X_train, Y_train, X_test)
    for i in range(preds_matrix.shape[0]):
            predicted_curve = list(preds_matrix[i, :])
            predictions[test_data['Device ID'].iloc[i]] = predicted_curve
    break

with open('all_devices_predicted_curves.pickle', 'wb') as handle:
    pickle.dump(predictions, handle)