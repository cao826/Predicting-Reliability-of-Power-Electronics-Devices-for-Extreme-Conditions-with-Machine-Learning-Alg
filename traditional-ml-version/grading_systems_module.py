import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit
import getpass
import pickle
import matplotlib.pyplot as plt
from sklearn import metrics

# utility functions

def get_actual_curve(device_id, data):
    """ Returns the measured curve of a device
    Arguments:
        device_id: string
    Returns:
        List containing the measured curve of the device
    """
    device_row = data[data['Device ID'] == device_id]
    actual_curve = list(device_row.iloc[0, -51:-1].values)
    return actual_curve

def get_true_failure_value(device_id, data):
    """ Returns the true failure value of the device
    Arguments:
        device_id: string
        data: pandas.DataFrame object. 
        Must be the modeling data provided in order to work
    Returns:
        Failure value of device
    """
    device_row = data[data['Device ID'] == device_id]
    failure_value = device_row['Fail/Not'].values[0]
    return failure_value

def get_predicted_points(device_id, predicted_curves_dict):
    """ Assuming the predicted curves comes in the 
    device_id:predicted_curve format of a dictionary,
    this function returns the predicted curve of a device
    Arguments:
        device_id: string
        predicted_curves_dict: dictionary
    Returns:
        Description of return. Does not need to include type
    """
    return predicted_curves_dict[device_id]


#Soft grading code

def get_point(index, x_data, y_data):
    """ Returns the x and y coordinates of the point specified by index
    Arguments:
        index: int
        x_data: list
        y_data: list
    Returns:
        tuple with x and y coordinates pulled from the lists
    """
    x_coordinate = x_data[index]
    y_coordinate = y_data[index]

    return (x_coordinate, y_coordinate)

def get_slope(x_1, y_1, x_0, y_0):
    """ Calculates slope
    Arguments:
        x_1: float
        y_1: float
        x_0: float
        y_0: float
    Returns:
        Calculated slope
    """
    slope = ( y_1 - y_0 ) / (x_1 - x_0)
    return slope

def slope_between_points(point1, point0, x_data, y_data):
    """ One line description of what function does
    Arguments:
        param1: parameter type
        param2: paremeter type
    Returns:
        Description of return. Does not need to include type
    """

    x0, y0 = get_point(point0, x_data, y_data)
    x1, y1 = get_point(point1, x_data, y_data)
    return get_slope(x1, y1, x0, y0)

def avg_slope(x_data, y_data):
    """ Calculates the average slope between chosen points.
    Arguments:
        x_data: list
        y_data: list
    Returns:
        Average slope
    """
    slopes = []
    for i in range(2,5):
        slope = slope_between_points((i+1), i, x_data, y_data)
        slopes.append(slope)
    return np.array(slopes).mean()#

# Hard grading code

def get_pass_window(labels_array):
    labels_max = labels_array.max()
    pass_window = abs(0.1 * labels_max)
    return pass_window

def check_point(predicted_point, label_point, pass_window):
    score = 0
    if (abs(predicted_point - label_point) < pass_window):
        score = 1
    return score

def check_predicted_curve(predictions_array, labels_array):
    pass_window = get_pass_window(labels_array)
    passed_points = 0
    for i in range(len(predictions_array)):
        predicted_point = predictions_array[i]
        label_point = labels_array[i]
        passed_points += check_point(predicted_point, label_point, pass_window)
    return passed_points

# Andrew's grading system -- needs to be renamed

def get_trendline(xs: list, ys: list, period=3):


    fit_xs = [] # Holds the fitted x points
    fit_ys = [] # Holds the fitted y points

    # Add an x, y fitted point at the first point in xs and ys
    fit_xs.append(xs[0])
    fit_ys.append(ys[0])

    # Add x, y fitted points for all data within xs and ys. Note, because this produces mean and median values, the first and last
    # data points in xs, ys need to be added to the fitted points as well
    for i in range(int(len(xs) / period)):
        fit_xs.append(np.median([xs[(i * period) + z] for z in range(period)])) # Middle of x values within the given period
        fit_ys.append(np.mean  ([ys[(i * period) + z] for z in range(period)])) # Mean of y values within the given period

    # Add an x, y fitted point at the last point in xs and ys
    fit_xs.append(xs[-1])
    fit_ys.append(ys[-1])

    return fit_xs, fit_ys

# Given a y amount, returns the x value at that y based on the list of x, y values
# Assumes that there can be multiple x values for any given y (unlike get_y_at_x())
def get_x_at_y(xs: list, ys: list, y: float, x_min: float=0.0, x_max: float=6.0e9, device_failure: bool=True):
    
    
    x   = [] # Holds the interpolated x at given y
    yi1 = 0  # Holds the y index that matches to the closest ys point to the y given
    yi2 = 1  # Holds the y index that matches to the second closest ys point to the y given

    # Check that the y isn't above or below the max and min y in xs and ys. If it is, return the min and max
    # x values respectively. If it isn't, interpolate the correct value for the x
    if y > np.max(ys):
        if device_failure:
            x.append(x_min)
        else:
            x.append(x_max)
    elif y < np.min(ys):
        if device_failure:
            x.append(x_max)
        else:
            x.append(x_min)
    else:

        # There can be multiple x values at a given y. To account for this, need to get all possible x values
        # and return them all. The code calling this function can then choose how to deal with that.

        # Find the y ranges between all points, gathering those points that have a range with the y we
        # are searching for
        point_ranges = [] # list of [[x1,y1], [x2,y2]] values which represent a range of y values between two points

        for yv in range(1, len(ys)):
            x1 = xs[yv - 1]
            x2 = xs[yv]
            y1 = ys[yv - 1]
            y2 = ys[yv]

            if y1 >= y2:
                if (y <= y1) and (y >= y2): # if y is within this range of y values
                    point_ranges.append([[x1, y1], [x2, y2]])
            elif y2 > y1:
                if (y <= y2) and (y >= y1): # if y is within this range of y values
                    point_ranges.append([[x1, y1], [x2, y2]])

        # For each range in point_ranges, calculate the fluence at the y given
        for i in range(len(point_ranges)):
            x1 = point_ranges[i][0][0]
            y1 = point_ranges[i][0][1]
            x2 = point_ranges[i][1][0]
            y2 = point_ranges[i][1][1]

            run   = x2 - x1 # x2 should always be bigger since x is fluence which is always increasing

            if run != 0:
                slope = (y2 - y1) / run # rise / run between two points
                b     = y1 - (slope * x1) # y = mx + b, b = y - mx
                if slope != 0:
                    x.append((y - b) / slope) # y = mx + b, x = (y - b) / m
            else: # slope is infinite (x1 and x2 are the same)
                x.append(x1)

    # Check if the list of x values is empty
    if x == []:
        print('WARNING: Couldn\'t find x at y point of ' + str(y) + '.')
        x.append(-1) # Still want to return a number, but set it to -1, which should never be a possible
                            # fluence value

    return x

'''
* This function is the same as the get_level_out_point in GenerateMLPipeline except it also returns the x point of failure.
* Returns the x and y points where the current (ys) has flattened out. (does not return if a device failed or not)
* xs containes fluence values and ys contains current values
* slice_height_percentage indicates the height of each horizontal slice on the x (fluence) by y (current) graph. The height is equal to the indicated percentage of the maximum current in said device. Whichever slice has the most fluence spanned by its data points is the slice where the device has flattened out (not necessarily failed)
* y_slice_fraction is used to determine what current value to take within the horizontal slice that spans the most fluence. The height of the slice is divided by this number and that value is taken as the current at point of 'failure' (flattening out of data, doesn't indicate actual failure)
'''
def get_level_out_point(xs: list, ys: list, slice_height_percentage=8, y_slice_fraction=2.5):
    
    level_off_point_y = 0 # The y (current) point where the x, y plot's current has leveled off
    level_off_point_x = 0  # The x (fluence) point where the x, y plot's fluence has leveled off
    max_ys            = np.max(ys) # Maximum current in the x, y plot
    min_ys            = np.min(ys) # Minimum current in the x, y plot
    height            = max_ys / 100 * slice_height_percentage # The height (in current) of each horizontal slice in the x, y plot
    total_slices      = int(max_ys / height) + 1 # An extra slice is added to ensure that values at the very top of the device plot aren't cut off.
    fluence_in_slices = [] # List of how much fluence is spanned by data points in each slice

    # Loop processing horizontal slices of device data
    for i in range(total_slices):
        y1 = min_ys + (height * i)       # Minimum allowed y value to be in current slice
        y2 = min_ys + (height * (i + 1)) # Maximum allowed y value to be in current slice.

        # Make sure that the bottom y value of the given slice isn't below the minumum current in ys or above the maximum current in ys
        if y1 < min_ys:
            y1 = min_ys
        elif y1 > max_ys:
            y1 = max_ys

        # Make sure that the top y value of the given slice isn't below the minumum current in ys or above the maximum current in ys
        if y2 > max_ys:
            y2 = max_ys
        elif y2 < min_ys:
            y2 = min_ys

        # Interpolate the fluence values at the top and bottom current values in the given slice
        # This allows a slice's fluence range to be calculated
        x1 = get_x_at_y(xs, ys, y1)[-1]
        x2 = get_x_at_y(xs, ys, y2)[-1]

        # Add the fluence range between the two fluence values to the fluence_in_slices along with its slice index
        fluence_in_slices.append([abs(x1 - x2), y1, y2])

    # Find the index of the fluence_in_slices with the highest number of data points
    most_index = 0

    for i in range(len(fluence_in_slices)):
        if fluence_in_slices[i][0] > fluence_in_slices[most_index][0]:
            most_index = i

    # Calculate the y value from within the slice with the highest fluence range
    y1            = fluence_in_slices[most_index][1] # Minimum allowed y value to be in current slice
    y2            = fluence_in_slices[most_index][2] # Maximum allowed y value to be in current slice.

    level_off_point_y = y1 + ((y2 - y1) / y_slice_fraction)
    level_off_point_x = get_x_at_y(xs, ys, level_off_point_y)[-1]

    return level_off_point_x, level_off_point_y

# Returns whether the device failed or not (based off of calculation not original categorization)
def is_failed(ys: list, flat_y: float):
    
    failed    = True
    threshold = np.max(ys) * 1/2

    if (flat_y > threshold):
        failed = False

    return failed

# bad curve detector

def get_max_current(fp_curve):
    """Returns the max current of the actual curve."""
    #print('enter get_max_current')
    return np.array(fp_curve).max()

def requires_null_grading(fpoint, max_current):
    """determines if a point is less than max_charge/10"""
    #print('enter requires_null_grading')
    null_grading_indicator = False
    null_grading_threshold = (max_current / 10)
    if fpoint <= null_grading_threshold:
        #print('NULL POINT'.format(fpoint, max_current))
        null_grading_indicator = True
    return null_grading_indicator

def is_bad_point_with_normal_grading(fpoint, apoint):
    """Uses a 5x window to determine if a point is good or bad"""
    #print('enter is_bad_point_with_normal_grading')
    bad_point_indicator = False
    if ((fpoint < (apoint / 5)) or (fpoint > (5 * apoint))):
        #print('BAD NORMAL POINT')
        bad_point_indicator = True
    return bad_point_indicator

def is_bad_point_with_null_grading(fpoint, apoint, max_current):
    """Uses a + or - max_charge/10 window to determine if a point is good or bad"""
    #print('enter is_bad_point_with_null_grading')
    bad_point_indicator = False
    if ((fpoint < (apoint - (max_current/10))) or (fpoint > (apoint + (max_current/10)))):
        #print('BAD NULL POINT')
        bad_point_indicator = True
    return bad_point_indicator

def determine_if_bad_curve(fcurve, acurve, cutoff, period, fluences):
    """
    Uses the functions above to count 
    the number of bad points and return a boolean 
    to indicate if the curve is bad
    """
    #print('enter determine_if_bad_curve')
    assert(len(fcurve) == len(acurve))
    fplist = fcurve[period:]
    aplist = acurve[period:]
    normal_null_lst = []
    good_bad_lst = []
    fluences_lst = fluences[period:]
    num_bad_points = 0
    bad_curve_indicator = False
    max_current = get_max_current(acurve)
    for i in range(len(fplist)):
        #print(i)
        fpoint = fplist[i]
        apoint = aplist[i]
        if (requires_null_grading(apoint, max_current)):
            normal_null_lst.append('Null')
            if is_bad_point_with_null_grading(fpoint, apoint, max_current):
                good_bad_lst.append('Bad')
                num_bad_points += 1
            else:
                good_bad_lst.append('Good')
        else:
            normal_null_lst.append('Normal')
            if is_bad_point_with_normal_grading(fpoint, apoint):
                good_bad_lst.append('Bad')
                num_bad_points += 1
            else:
                good_bad_lst.append('Good')
    if num_bad_points >= cutoff:
        bad_curve_indicator = True
        
    grade_df = pd.DataFrame({'Fluence':fluences_lst,
                             'Actual Point':aplist, 
                             'Predicted Point': fplist, 
                             'Normal/Null':normal_null_lst,
                             'Good/Bad Predicted Point':good_bad_lst})
    
    return bad_curve_indicator, grade_df

def check_preconditions_for_moving_average(curve, N):
    assert(type(curve) == list)
    assert(N>0)
    assert(len(curve) > N)
    
def pad_w_nan(avg_curve, period):
    padded_avg_curve = ([np.NaN]*period) + avg_curve
    return padded_avg_curve

def moving_average_of(curve, N):
    check_preconditions_for_moving_average(curve, N)
    M = len(curve)
    avg_curve = []
    for k in range(N, M):
        avg = 0
        for l in range(1, (N+1)):
            avg += (curve[(k - l)] / N)
        avg_curve.append(avg)
    padded_avg_curve = pad_w_nan(avg_curve, N)
    return padded_avg_curve   

