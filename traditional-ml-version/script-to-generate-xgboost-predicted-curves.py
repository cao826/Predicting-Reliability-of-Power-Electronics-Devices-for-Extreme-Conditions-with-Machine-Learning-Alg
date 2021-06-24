import pandas as pd
import numpy as np
import argparse as arg
import pickle
import matplotlib.pyplot as plt
from sklearn import metrics
from tqdm import tqdm
import gradboost_module as gmod
import grading_systems_module as grad

parser = arg.ArgumentParser(description="Produces predicted points with LSTM based encoder-decoder models.")
parser.add_argument('Path',
        metavar='path',
        type=str,
        help='Path to the data for deep learning')

args = parser.parse_args()
path = args.Path

data_w_labels = pd.read_csv(path)
fluences = [float(column.split(' ')[-2]) for column in data_w_labels.columns[-51:-1]]


predicted_curves = dict()
for i in range(1, 25):
    test_data, train_data = gmod.get_fold_k(i, data_w_labels)
    X_train = train_data.iloc[:, 1:-51].values
    Y_train = train_data.iloc[:, -51:-1].values
    X_test = test_data.iloc[:, 1:-51].values
    preds_matrix = gmod.xgboostmult(X_train, Y_train, X_test)
    for i in range(preds_matrix.shape[0]):
            predicted_curve = list(preds_matrix[i, :])
            predicted_curves[test_data['Device ID'].iloc[i]] = predicted_curve

with open('all_devices_predicted_curves.pickle', 'wb') as handle:
    pickle.dump(predicted_curves, handle)

bad_count = 0
good_count = 0
bad_curve_dict = dict()
for device in predicted_curves.keys():
    #true_failure_val = get_true_failure_value(device)
    actual_curve = grad.get_actual_curve(device,data_w_labels)
    predicted_points = predicted_curves[device]
    fcurve = grad.moving_average_of(predicted_points, 3)
    bad_curve_indicator, bad_curve_df = grad.determine_if_bad_curve(fcurve, actual_curve, 15, 3, fluences)
    
    if bad_curve_indicator == True:
        bad_count += 1
        bad_curve_dict[device] = 'Bad'
    else:
        good_count += 1
        bad_curve_dict[device] = 'Good'

print('bad curves: {}'.format(bad_count))
print('good curves: {}'.format(good_count))
print('Total: {}'.format(good_count + bad_count))

# umm

ture_vals = data_w_labels['Fail/Not'].values


# soft grading results

device_ids = []
fail_status = []

for device in predicted_curves.keys():
    predicted_points = predicted_curves[device]
    slope = grad.avg_slope(fluences, predicted_points)
    device_ids.append(device)
    if (slope < 0):
        fail_status.append('Fail')
    else:
        fail_status.append('Pass')

soft_grading_df = pd.DataFrame({'Device ID':device_ids, 'Soft Grading':fail_status})
print(soft_grading_df)

soft_grade_check = data_w_labels.merge(soft_grading_df, on=('Device ID'), how='left')

print(soft_grade_check['Fail/Not'])
print(soft_grade_check['Soft Grading'])

print("soft grading: {}".format(metrics.accuracy_score(soft_grade_check['Fail/Not'],soft_grade_check["Soft Grading"])))

# hard grading 

device_ids = []
curve_status = []
good_count = 0

for device in predicted_curves.keys():
    predicted_points = predicted_curves[device]
    actual_curve = grad.get_actual_curve(device, data_w_labels)
    actual_curve = np.array(actual_curve)
    passed_points = grad.check_predicted_curve(predicted_points, actual_curve)
    device_ids.append(device)
    if (passed_points > 35):
        curve_status.append('Good')
        good_count += 1
    else:
        curve_status.append('Bad')


print(good_count)

# andrew's grading 

device_ids = []
predictions = []
for device in predicted_curves.keys():
    device_ids.append(device)
    predicted_points = predicted_curves[device]
    fit_xs, fit_ys = grad.get_trendline(fluences, predicted_points)
    flat_x, flat_y    = grad.get_level_out_point(fit_xs, fit_ys)
    failed            = grad.is_failed(fit_ys, flat_y)
    if failed:
        predictions.append('Fail')
    else:
        predictions.append('Pass')

andrew_grading_df = pd.DataFrame({'Device ID':device_ids,  "Andrew's Grading":predictions})
andrew_grading_check = data_w_labels.merge(andrew_grading_df, on=('Device ID'), how='left')

print("andrew grading: {}".format(metrics.accuracy_score(andrew_grading_check['Fail/Not'],andrew_grading_check["Andrew's Grading"])))
