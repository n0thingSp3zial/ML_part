import json
import os
import numpy as np
import matplotlib.pyplot as plt

GESTURE_SIZE = 256
data_for_fuzzy = []

#======================== Very Smart Cut ===========================

def less_then_counter(array, end_index, eps):
    index = 0
    while index < end_index and array[index] < eps:
        index += 1
    
    return index

def smart_cut(gesture, max_gesture_len=GESTURE_SIZE):
    res = gesture
    delta = gesture.size - max_gesture_len
    eps = np.amax(gesture) * 0.12
    
    if delta > 0:
        left_delete_index = less_then_counter(np.abs(gesture), delta // 2, eps)
        res = gesture[left_delete_index:-(delta-left_delete_index)]
    elif delta < 0:
        delta = -delta
        res = np.pad(gesture, (delta // 2, delta - delta // 2))
    
    return res


def calc_features(main_dir):
    #======================== Open Files ===========================

    gestures_dir = sorted(os.listdir(main_dir))

    features_array = []
    dtw_distance_array = []
    

    for gesture_file_dir in gestures_dir:
        if gesture_file_dir == '.DS_Store':
            continue
            
        gesture_file = open(main_dir + gesture_file_dir, mode='r')
        gesture_in_json = gesture_file.read()
        gesture_file.close()
        time_points = json.loads(gesture_in_json)                   # array of dict
        
        x_acc = smart_cut(np.array([d['x_acc'] for d in time_points]))
        y_acc = smart_cut(np.array([d['y_acc'] for d in time_points]))
        z_acc = smart_cut(np.array([d['z_acc'] for d in time_points]))
        
        a_rot = smart_cut(np.array([d['a_rot'] for d in time_points]))
        b_rot = smart_cut(np.array([d['b_rot'] for d in time_points]))
        g_rot = smart_cut(np.array([d['g_rot'] for d in time_points]))

        #======================== Median Filters ===========================
        
        cur_features = []
        cur_file_time_series = []
        for cur_time_seris in [x_acc, y_acc, z_acc, a_rot, b_rot, g_rot]:
            med_cur_time_seris = [(x + y + z) / 3 for x, y, z in
                zip(np.pad(cur_time_seris, (0, 2)),
                    np.pad(cur_time_seris, 1),
                    np.pad(cur_time_seris, (2, 0)))][1:-1]
            
            #plt.scatter(np.arange(GESTURE_SIZE), med_cur_time_seris, s=10)
            #plt.grid(True)
            #plt.show()
            
            #===================== Fast Fourier Transform =======================

            gesture_after_fft = np.fft.fft(med_cur_time_seris)
            #plt.scatter(np.arange(GESTURE_SIZE), gesture_after_fft, s=10)
            #plt.grid(True)
            #plt.show()

            #===================== Wavelet Transform ============================

            import pywt

            (cA, cD) = pywt.dwt(gesture_after_fft, 'haar')

            #====================== Z-score Normalization =======================

            import scipy.stats as stats

            zscores_cA = stats.zscore(cA)
            zscores_cD = stats.zscore(cD)

            #======================= Calculating Features =======================

            max_cA = np.amax(zscores_cA)
            min_cA = np.amin(zscores_cA)
            avg_cA = np.average(zscores_cA)
            std_cA = np.std(zscores_cA)

            max_cD = np.amax(zscores_cD)
            min_cD = np.amin(zscores_cD)
            avg_cD = np.average(zscores_cD)
            std_cD = np.std(zscores_cD)
            
            #======================== Adding Features ========================
            
            cur_features.extend([max_cA, min_cA, avg_cA, std_cA,
                                max_cD, min_cD, avg_cD, std_cD])
                                
            #======================= Adding Time Series =======================
            
            cur_file_time_series.append(med_cur_time_seris)

        features_array.append(cur_features)
        data_for_fuzzy.append(np.array(cur_file_time_series).T)
    return np.absolute(np.array(features_array))

#===================== Calculating DTW-distance =====================

from dtaidistance import dtw

#d = dtw.distance_fast(x_acc1, x_acc1)

#=========================== Fuzzy Logic ============================

from FuzzyClassifier import FuzzyClassifier


fuzzy = FuzzyClassifier(m = 1.5, k = 0.8)
calc_features('/Users/n0thingsp3zial/Desktop/иит/курсач/gestures/train/')
print(len(data_for_fuzzy))

fuzzy.train(data_for_fuzzy)                # data - 3d array: 1st_d - files; 2nd_d - time_points; 3rd_d - 3 axis of acc
for record in data_for_fuzzy:               # record - 1 file
    res = fuzzy.classify(record)
    print('yes' if res > 0.5 else 'no')
    
calc_features('/Users/n0thingsp3zial/Desktop/иит/курсач/gestures/test/')
print('testing fuzzy:')

for record in data_for_fuzzy:               # record - 1 file
    res = fuzzy.classify(record)
    print('yes' if res > 0.5 else 'no')

#=============================== SVM ================================
'''
from OneClassSVM import OneClassSVM

calc_features('/Users/n0thingsp3zial/Desktop/иит/курсач/gestures/train/')
print(len(data_for_fuzzy))
svm = OneClassSVM(nu = 2.0 / len(data_for_fuzzy))
svm.train(data_for_fuzzy)

for record in data_for_fuzzy:               # record - 1 file
    res = svm.classify(record)
    #print('yes' if res > 0.5 else 'no')
    print(res)

data_for_fuzzy = []
calc_features('/Users/n0thingsp3zial/Desktop/иит/курсач/gestures/test/')
print('testing svm:')

for record in data_for_fuzzy:               # record - 1 file
    res = svm.classify(record)
    #print('yes' if res > 0.5 else 'no')
    print(res)
'''
