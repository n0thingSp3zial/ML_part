import numpy as np
import pywt
import scipy.signal as signal
import scipy.stats as stats
from statistics import variance

#================================ Map Functions =================================

def map_funcs(gesture, func_list, kwargs):
    '''
    Apply list of functions to given gesture with given parameters
    '''
    res = gesture
    for i, func in enumerate(func_list):
        res = func(res, *kwargs[i])
    return res

#============================= Frequency Reduction ==============================

def freq_red(gesture, cur_freq, expected_freq):
    res = []
    if (cur_freq <= expected_freq):
        k = expected_freq // cur_freq
        for cur_axis in gesture:
            tmp = []
            for cur_val, next_val in zip(cur_axis[:-1], cur_axis[1:]):
                tmp.extend(np.linspace(cur_val, next_val, num=k, endpoint=False))
            tmp.append(cur_axis[-1])
            res.append(tmp)
    else:
        k = cur_freq // expected_freq
        for cur_axis in gesture:
            res.append(cur_axis[::k])
    return res
    
#================================= Running Mean =================================

def running_mean(gesture, window_size):
    '''
    Returns smoother array
    '''
    if len(gesture[0]) > window_size:
        res = []
        for cur_axis in gesture:
            cumsum = np.cumsum(np.insert(cur_axis, 0, 0))
            res.append((cumsum[window_size:] - cumsum[:-window_size]) / float(window_size))
        return np.stack(res)
    else:
        return gesture

#================================ Median Filter =================================

def median_filter(gesture, window_size):
    '''
    Returns smoother array
    '''
    if len(gesture[0]) > window_size:
        res = []
        for cur_axis in gesture:
            res.append(signal.medfilt(cur_axis, kernel_size=window_size))
        return np.stack(res)
    else:
        return gesture
        
#============================= Butterworth Filter ===============================

def butter_filter(gesture, wp, ws, gpass, gstop, fs):
    '''
    Returns smoother array
    '''
    N, Wn = signal.buttord(wp=wp, ws=ws, gpass=gpass, gstop=gstop)
    sos = signal.butter(N, Wn, btype='lp', output='sos')
    res = []
    for cur_axis in gesture:
        res.append(signal.sosfilt(sos, cur_axis))
    return np.stack(res)
    
#============================ Savitzky-Golay Filter =============================

def sg_filter(gesture, window_size, polyorder):
    '''
    Returns smoother array
    '''
    if len(gesture[0]) > window_size:
        res = []
        for cur_axis in gesture:
            res.append(signal.savgol_filter(cur_axis, window_length=window_size, polyorder=polyorder))
        return np.stack(res)
    else:
        return gesture
        
#================================= Cubic Spline =================================

def cspline(gesture, lamb):
    '''
    Returns smoother array
    '''
    res = []
    for cur_axis in gesture:
        res.append(signal.cspline1d(cur_axis, lamb=lamb))
    return np.stack(res)
    
#============================== Wavelet Transform ===============================

def wavelet_transform(gesture, times_to_apply, type):
    '''
    Returns approximation and detalisation data combined for every axis of data
    '''
    res = []
    for cur_axis in gesture:
        aprox = cur_axis
        for _ in range(times_to_apply):
            aprox, detail = pywt.dwt(aprox, type)
        #for _ in range(times_to_apply):
            #aprox = pywt.idwt(aprox, None, type)
        res.append(aprox)
    return np.stack(res)
    
#============================= From Axes To Module ==============================

def from_axes_to_modules(gesture, add_new_axis):
    modules = np.sqrt(np.square(gesture.reshape((-1, 3, gesture.shape[-1]))).sum(1))
    if add_new_axis:
        return np.vstack([gesture, modules])
    else:
        return modules

#=============================== Extract Features ===============================

def feature_extract(gesture):
    '''
    Returns extracted features from gesture
    '''
    res = []
    for cur_axis in gesture:
        res.extend([np.mean(cur_axis),
                    np.min(cur_axis),
                    np.max(cur_axis),
                    np.std(cur_axis),
                    np.percentile(cur_axis, 87),
                    stats.kurtosis(cur_axis),
                    stats.skew(cur_axis),
                    stats.expectile(cur_axis),
                    stats.iqr(cur_axis),
                    stats.sem(cur_axis),
                    ])
    return np.array(res)
    
#================================ Very Smart Cut ================================

def less_then_counter(array, end_index, eps):
    '''
    Returns index of the first element, that less then eps
    '''
    index = 0
    while index < end_index and array[index] < eps:
        index += 1
    
    return index

def smart_cut(gesture, max_gesture_len):
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
