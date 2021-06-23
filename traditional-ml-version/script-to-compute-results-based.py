import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import grading_systems_module as grad
from sklearn import metrics

predicted_curves_path = "anon-pred-curves.pickle"

with open(predicted_curves_path, 'rb') as f:
    orig_predicted_curves = pickle.load(f)

data_path = "/Users/calchuchesta/Box/Prime technical folder ML and AI work/Carlos's Folder/Summer_2021/data/modeling-data.csv"

new_data = pd.read_csv(data_path)
fluences = [float(column.split(' ')[-2]) for column in new_data.columns[-51:-1]]

bad_count = 0
good_count = 0
bad_curve_dict = dict()
for device in orig_predicted_curves.keys():
    #true_failure_val = get_true_failure_value(device)
    actual_curve = grad.get_actual_curve(device,new_data)
    predicted_points = orig_predicted_curves[device]
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

ture_vals = new_data['Fail/Not'].values


# soft grading results

device_ids = []
fail_status = []

for device in orig_predicted_curves.keys():
    predicted_points = orig_predicted_curves[device]
    slope = grad.avg_slope(fluences, predicted_points)
    device_ids.append(device)
    if (slope < 0):
        fail_status.append('Fail')
    else:
        fail_status.append('Pass')

soft_grading_df = pd.DataFrame({'Device ID':device_ids, 'Soft Grading':fail_status})

soft_grade_check = new_data.merge(soft_grading_df, on=('Device ID'), how='left')

print("soft grading: {}".format(metrics.accuracy_score(soft_grade_check['Fail/Not'],soft_grade_check["Soft Grading"])))

# hard grading 

device_ids = []
curve_status = []
good_count = 0

for device in orig_predicted_curves.keys():
    predicted_points = orig_predicted_curves[device]
    actual_curve = grad.get_actual_curve(device, new_data)
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
for device in orig_predicted_curves.keys():
    device_ids.append(device)
    predicted_points = orig_predicted_curves[device]
    fit_xs, fit_ys = grad.get_trendline(fluences, predicted_points)
    flat_x, flat_y    = grad.get_level_out_point(fit_xs, fit_ys)
    failed            = grad.is_failed(fit_ys, flat_y)
    if failed:
        predictions.append('Fail')
    else:
        predictions.append('Pass')

andrew_grading_df = pd.DataFrame({'Device ID':device_ids,  "Andrew's Grading":predictions})
andrew_grading_check = new_data.merge(andrew_grading_df, on=('Device ID'), how='left')

print("andrew grading: {}".format(metrics.accuracy_score(andrew_grading_check['Fail/Not'],andrew_grading_check["Andrew's Grading"])))
