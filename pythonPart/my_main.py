import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from sklearn.metrics import roc_curve, auc

GESTURE_SIZE = 256

#=============================== Time Series Class ==============================

class TimeSeries:
    '''
    acc: np.array (time series of accelerometer)
    rot: np.array (time series of gyroscope)
    '''
    def __init__(self, x_acc, y_acc, z_acc, a_rot, b_rot, g_rot):
        self.x_acc = x_acc
        self.y_acc = y_acc
        self.z_acc = z_acc
        self.a_rot = a_rot
        self.b_rot = b_rot
        self.g_rot = g_rot
        
    def __len__(self):
        '''
        Returns length of time series
        '''
        if self.x_acc.size == self.y_acc.size == self.z_acc.size == self.a_rot.size == self.b_rot.size == self.g_rot.size:
            return self.x_acc.size
        else:
            return -1
            
    def stack_data(self, sensor='acc+gyro'):
        '''
        Returns 2d-array: 1st_d - different axis; 2nd_d - time_points;
        '''
        if sensor == 'acc+gyro':
            return np.stack([self.x_acc, self.y_acc, self.z_acc, self.a_rot, self.b_rot, self.g_rot])
        elif sensor == 'acc':
            return np.stack([self.x_acc, self.y_acc, self.z_acc])
        elif sensor == 'gyro':
            return np.stack([self.a_rot, self.b_rot, self.g_rot])
        else:
            print('Error: non-existent sensor type')

#================================== User Class ==================================

class UserClass:
    '''
    username: string
    time_series: [class TimeSeries], that contains accelerometer and gyroscope data of user
    features: np.array (features, that are used in classification)
    '''
    def __init__(self, username):
        self.username = username
        self.time_series_cat = []
        self.time_series_inf = []
        self.time_series_12 = []
        self.time_series_star = []
        
    def __len__(self):
        '''
        Returns amount of available gesture types
        '''
        len = 0
        if self.time_series_cat:
            len += 1
        if self.time_series_inf:
            len += 1
        if self.time_series_12:
            len += 1
        if self.time_series_star:
            len += 1
        return len
        
    def times_series_len(self, type):
        if type == 'cat':
            return(len(self.time_series_cat))
        if type == 'inf':
            return(len(self.time_series_inf))
        if type == '12':
            return(len(self.time_series_12))
        if type == 'star':
            return(len(self.time_series_star))
        
    def available_gestures(self):
        '''
        Returns an array of names of available gesture types
        '''
        ag = []
        if self.time_series_cat:
            ag.append('cat')
        if self.time_series_inf:
            ag.append('inf')
        if self.time_series_12:
            ag.append('12')
        if self.time_series_star:
            ag.append('star')
        return ag
        
    def get_time_series(self, type, transformer=[lambda x: x], sensor='acc+gyro'):
        '''
        Returns 3d-array: 1st_d - files; 2nd_d - time_points; 3rd_d - 3 axis of sensor
        '''
        if type == 'star':
            return [map_funcs(cur_ts.stack_data(sensor), transformer).T for cur_ts in self.time_series_star]
        elif type == 'inf':
            return [map_funcs(cur_ts.stack_data(sensor), transformer).T for cur_ts in self.time_series_inf]
        elif type == '12':
            return [map_funcs(cur_ts.stack_data(sensor), transformer).T for cur_ts in self.time_series_12]
        elif type == 'cat':
            return [map_funcs(cur_ts.stack_data(sensor), transformer).T for cur_ts in self.time_series_cat]
        else:
            print('Error: non-existent gesture type')
            
#================================ Map Functions =================================

def map_funcs(gesture, func_list):
    '''
    Apply list of functions to given gesture
    '''
    res = gesture
    for func in func_list:
        res = func(res)
    return res

#================================ Very Smart Cut ================================

def less_then_counter(array, end_index, eps):
    '''
    Returns index of the first element, that less then eps
    '''
    index = 0
    while index < end_index and array[index] < eps:
        index += 1
    
    return index

def smart_cut(gesture, max_gesture_len=GESTURE_SIZE):
    '''
    Returns time series reduced to a given length
    '''
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
    
def transformer_smart_cut(gesture):
    res = []
    for cur_axis in gesture:
        res.append(smart_cut(cur_axis))
    return np.stack(res)
    
#================================ Median Filter =================================

def median_filter(gesture):
    '''
    Return smoother array
    '''
    res = []
    for cur_axis in gesture:
        res.append(np.array([(x + y + z) / 3 for x, y, z in
            zip(np.pad(cur_axis, (0, 2)), np.pad(cur_axis, 1), np.pad(cur_axis, (2, 0)))][1:-1]))
    return np.stack(res)
    
#=============================== Extract Features ===============================

def feature_extract(time_series):
    '''
    Returns extracted features from given TimeSeries object
    '''
    return np.array([0, 0.5, 1])
    
#================================ Reading Files =================================

def create_gesture_base(gestures_directory, mode):
    '''
    Returns np.array of UserClass objects
    '''
    gesture_base = []

    for cur_user in os.listdir(gestures_directory):
        if cur_user == '.DS_Store' or cur_user == '_Images' or cur_user.endswith('.gz'):
            continue
        
        cur_user_obj = UserClass(cur_user)
        
        user_directory = os.path.join(gestures_directory, cur_user)
        for cur_gesture_type in os.listdir(user_directory):
            if cur_gesture_type == '.DS_Store':
                continue

            train_dir = os.path.join(user_directory, cur_gesture_type, 'train')
            test_dir = os.path.join(user_directory, cur_gesture_type, 'test')
            attack_dir = os.path.join(user_directory, cur_gesture_type, 'attack')
            
            if mode == 'train':
                cur_dir = train_dir
            elif mode == 'test':
                cur_dir = test_dir
            else:
                cur_dir = attack_dir
                
            for cur_file in os.listdir(cur_dir):
                if cur_file == '.DS_Store':
                    continue
                
                with open(os.path.join(cur_dir, cur_file), 'r') as file:
                    file_string = file.read()
                    file_json_obj = json.loads(file_string)
                    file_data = file_json_obj["params"]["data"]
                    
                    x_acc, y_acc, z_acc = np.array([x["acceleration"] for x in file_data]).T
                    a_rot, b_rot, g_rot = np.array([x["rotation"] for x in file_data]).T
                    frequency = file_json_obj["params"]["frequency"]
                    
                    cur_ts = TimeSeries(x_acc, y_acc, z_acc, a_rot, b_rot, g_rot)
                    
                    if cur_gesture_type == 'star':
                        cur_user_obj.time_series_star.append(cur_ts)
                    elif cur_gesture_type == '12':
                        cur_user_obj.time_series_12.append(cur_ts)
                    elif cur_gesture_type == 'cat':
                        cur_user_obj.time_series_cat.append(cur_ts)
                    elif cur_gesture_type == 'inf':
                        cur_user_obj.time_series_inf.append(cur_ts)
                    else:
                        print('error')
        gesture_base.append(cur_user_obj)
    return np.array(gesture_base)

#================================================================================

#================================================================================

train_gesture_base = create_gesture_base('/Users/n0thingsp3zial/Desktop/иит/курсач/Александр Ковальчук/Итоговые материалы/Данные/MoBe_Data/', 'train')

test_gesture_base = create_gesture_base('/Users/n0thingsp3zial/Desktop/иит/курсач/Александр Ковальчук/Итоговые материалы/Данные/MoBe_Data/', 'test')

from FuzzyClassifier import FuzzyClassifier
from OneClassSVM import OneClassSVM
from KNN import KNN

print()
from math import sqrt


for cur_user in train_gesture_base:
    available_gestures = cur_user.available_gestures()
    
    print('Starting training for user:', cur_user.username)
    
    for cur_gesture in available_gestures:
        X_train = cur_user.get_time_series(cur_gesture, transformer=[median_filter])
        X_test = []
        y_test = []
        for attacker in test_gesture_base:
            if cur_gesture not in attacker.available_gestures():
                continue
            if attacker.username == cur_user.username:
                X_test.extend(attacker.get_time_series(cur_gesture, transformer=[median_filter]))
                y_test.extend([1 for _ in range(attacker.times_series_len(cur_gesture))])
            else:
                X_test.extend(attacker.get_time_series(cur_gesture, transformer=[median_filter]))
                y_test.extend([0 for _ in range(attacker.times_series_len(cur_gesture))])
        
        #fuzzy = FuzzyClassifier(m = 1.5, k = 0.8)
        #fuzzy = OneClassSVM(nu = 2.0 / len(X_train))
        fuzzy = KNN(k = sqrt(len(X_train)))
        
        print(f'\tGesture: \'{cur_gesture}\'...')
        fuzzy.train(X_train)
        print('\t\tTraining is finished!')
        
        res = []
        print('\t\tStart testing...')
        ten_perc = len(X_test) // 10
        i = 0
        print(f'\r\t\t\t{(i // ten_perc) * 10}%', end='')
        for record in X_test:
            res.append(fuzzy.classify(record))
            i += 1
            if i % ten_perc == 0:
                print(f'\r\t\t\t{(i // ten_perc) * 10}%', end='')
        print('\r\t\tTesting is finished')
        
        fpr, tpr, thresholds = roc_curve(y_test, res)
        roc_auc = auc(fpr, tpr)
        #pl.plot(fpr, tpr, 'b', label = "Fuzzy - {0:.5f}".format(roc_auc), lw = 1)
        print(fpr)
        print(tpr)
        #print(thresholds)
        print(roc_auc)
    







'''
dana = train_gesture_base[0]

print(dana.username)
print(dana.available_gestures())
ts = dana.get_time_series('12', transformer=transformer_smart_cut)

tmp_ts = ts[0].T

print(tmp_ts.shape)

plt.scatter(np.arange(tmp_ts[0].size), tmp_ts[0], s=10, c='#04d9ff')
plt.scatter(np.arange(tmp_ts[1].size), tmp_ts[1], s=10, c='#bc13fe')
plt.scatter(np.arange(tmp_ts[2].size), tmp_ts[2], s=10, c='#5555ff')


plt.grid(True)
plt.show()
'''
