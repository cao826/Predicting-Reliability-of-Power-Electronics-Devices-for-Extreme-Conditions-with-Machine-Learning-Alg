import pandas as pd
import numpy as np
from   sklearn.ensemble        import GradientBoostingRegressor
from tqdm import tqdm

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

def xgboost():
    """ Returns a GradientBoosting model instance with the desired hyperparameters for this project.
    Arguments: None
    Returns:
        GradientBoosting Model instance
    """
    return GradientBoostingRegressor(n_estimators=200,
                                     learning_rate=0.05,
                                     max_depth=3,
                                     max_features=None,
                                     random_state=None)

def xgboostmult(X_train, Y_train, X_eval):
    """ One line description of what function does
    Arguments:
        param1: parameter type
        param2: paremeter type
    Returns:
        Description of return. Does not need to include type
    """
    preds_lst = []
    num_points_to_predict = Y_train.shape[1]
    for i in tqdm(range(num_points_to_predict)):
        model = xgboost()
        model.fit(X_train, Y_train[:, i])
        preds = model.predict(X_eval)
        #print(preds)
        preds_lst.append(preds)
    return np.stack(preds_lst, axis=1)

