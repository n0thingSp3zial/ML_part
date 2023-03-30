from transforms import map_funcs
import numpy as np
import os
import json
import fnmatch

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
            
    def stack_data(self, sensor):
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
        
    def get_time_series(self, type, transformer, sensor, kwargs):
        '''
        Returns 3d-array: 1st_d - files; 2nd_d - time_points; 3rd_d - 3 axis of sensor
        '''
        if type == 'star':
            return [map_funcs(cur_ts.stack_data(sensor), transformer, kwargs).T.tolist() for cur_ts in self.time_series_star]
        elif type == 'inf':
            return [map_funcs(cur_ts.stack_data(sensor), transformer, kwargs).T.tolist() for cur_ts in self.time_series_inf]
        elif type == '12':
            return [map_funcs(cur_ts.stack_data(sensor), transformer, kwargs).T.tolist() for cur_ts in self.time_series_12]
        elif type == 'cat':
            return [map_funcs(cur_ts.stack_data(sensor), transformer, kwargs).T.tolist() for cur_ts in self.time_series_cat]
        else:
            print('Error: non-existent gesture type')


#============================== Creating DataBase ===============================


def create_gesture_base(gestures_directory, mode):
    '''
    Returns np.array of UserClass objects
    '''
    gesture_base = []

    for cur_user in sorted(os.listdir(gestures_directory)):
        if cur_user == '.DS_Store' or cur_user == '_Images' or cur_user == '_gzs': # or cur_user.endswith('.gz')
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
                    
                cur_ts = TimeSeries([], [], [], [], [], [])
                
                if fnmatch.fnmatch(cur_file, '*.txt'):
                    with open(os.path.join(cur_dir, cur_file), 'r') as file:
                        file_string = file.read()
                        file_json_obj = json.loads(file_string)
                        file_data = file_json_obj['params']['data']
                        
                        x_acc, y_acc, z_acc = np.array([d['acceleration'] for d in file_data]).T
                        a_rot, b_rot, g_rot = np.array([d['rotation'] for d in file_data]).T
                        #frequency = file_json_obj["params"]["frequency"]
                        
                        cur_ts = TimeSeries(x_acc, y_acc, z_acc, a_rot, b_rot, g_rot)
                elif fnmatch.fnmatch(cur_file, '*.json'):
                    with open(os.path.join(cur_dir, cur_file), 'r') as file:
                        file_string = file.read()
                        file_json_obj = json.loads(file_string)
                                                
                        x_acc = np.array([d['x_acc'] for d in file_json_obj])
                        y_acc = np.array([d['y_acc'] for d in file_json_obj])
                        z_acc = np.array([d['z_acc'] for d in file_json_obj])
                        
                        if 'x_rot' in file_json_obj[0]:
                            a_rot = np.array([d['x_rot'] for d in file_json_obj])
                            b_rot = np.array([d['y_rot'] for d in file_json_obj])
                            g_rot = np.array([d['z_rot'] for d in file_json_obj])
                        else:
                            a_rot = np.array([d['a_rot'] for d in file_json_obj])
                            b_rot = np.array([d['b_rot'] for d in file_json_obj])
                            g_rot = np.array([d['g_rot'] for d in file_json_obj])
                        
                        #frequency = 100
                        cur_ts = TimeSeries(x_acc, y_acc, z_acc, a_rot, b_rot, g_rot)
                else:
                    print('Error: unknown type of file')
                                        
                if cur_gesture_type == 'star':
                    cur_user_obj.time_series_star.append(cur_ts)
                elif cur_gesture_type == '12':
                    cur_user_obj.time_series_12.append(cur_ts)
                elif cur_gesture_type == 'cat':
                    cur_user_obj.time_series_cat.append(cur_ts)
                elif cur_gesture_type == 'inf':
                    cur_user_obj.time_series_inf.append(cur_ts)
                else:
                    print('Error: unknown gesture type')
                            
        gesture_base.append(cur_user_obj)
    return np.array(gesture_base)
