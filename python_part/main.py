import numpy as np
from sklearn.metrics import roc_curve, auc
from math import sqrt
from transforms import freq_red, running_mean, median_filter, butter_filter, sg_filter, cspline, wavelet_transform, from_axes_to_modules, feature_extract
from GaussKernels import GaussKernel, DTWGaussKernel
from data_related import UserClass, create_gesture_base
from FuzzyClassifier import FuzzyClassifier
from OneClassSVM import OneClassSVM
from KNN import KNN

#================================ Hyperparameters ===============================


MY_DATA_USERNAMES = ['achikin', 'akirillov', 'apichkur', 'khabud', 'mkhrebtov', 'ochikin', 'opichkur', 'pchikin', 'tchikina', 'tnikulina']
DETAILED_OUTPUT = False

# def running_mean(gesture, window_size):
# def median_filter(gesture, window_size):
# def butter_filter(wp=0.34, ws=0.60, gpass=1, gstop=30, fs=100):
# def sg_filter(gesture, window_size, polyorder):
# def cspline(gesture, lamb):

# def from_axes_to_modules(gesture, add_new_axis):
# def wavelet_transform(gesture, times_to_apply, type):
# def freq_red(gesture, cur_freq, expected_freq):
# def feature_extract(gesture):

SENSORS = 'acc+gyro'
TRANSFORMS = [from_axes_to_modules, butter_filter, wavelet_transform, running_mean, wavelet_transform, running_mean, wavelet_transform, cspline]
KWARGS = [[True], [0.17, 0.3, 1, 30, None], [1, 'haar'], [9], [1, 'haar'], [7], [1, 'haar'], [3]]

KERNEL = GaussKernel if TRANSFORMS[-1] == feature_extract else DTWGaussKernel

#================================= Testing Zone =================================


train_gesture_base = create_gesture_base('/Users/n0thingsp3zial/Desktop/иит/курсач/Александр Ковальчук/Итоговые материалы/Данные/MoBe_Data/', 'train')

test_gesture_base = create_gesture_base('/Users/n0thingsp3zial/Desktop/иит/курсач/Александр Ковальчук/Итоговые материалы/Данные/MoBe_Data/', 'test')

fuzzy_res = []
svm_res = []
knn_res = []

for cur_user in train_gesture_base:

    if cur_user.username not in MY_DATA_USERNAMES:       # !!!!!!!!!!!!!!!!!
        continue
        
    available_gestures = cur_user.available_gestures()
    
    if DETAILED_OUTPUT:
        print('\n\nStarting testing for user:', cur_user.username)
    
    for cur_gesture in available_gestures:
        
        X_train = cur_user.get_time_series(cur_gesture, TRANSFORMS, SENSORS, KWARGS)
        X_test = []
        y_test = []
        for attacker in test_gesture_base:
        
            #if attacker.username in MY_DATA_USERNAMES:       # !!!!!!!!!!!!!!!!!
                #continue
        
            if cur_gesture not in attacker.available_gestures():
                continue
            if attacker.username == cur_user.username:
                X_test.extend(attacker.get_time_series(cur_gesture, TRANSFORMS, SENSORS, KWARGS))
                y_test.extend([1 for _ in range(attacker.times_series_len(cur_gesture))])
            else:
                X_test.extend(attacker.get_time_series(cur_gesture, TRANSFORMS, SENSORS, KWARGS))
                y_test.extend([0 for _ in range(attacker.times_series_len(cur_gesture))])
        
        fuzzy = FuzzyClassifier(kernel=KERNEL('Fuzzy'), m = 1.5, k = 0.8)
        svm = OneClassSVM(kernel=KERNEL('SVM'), nu = 2.0 / len(X_train))
        knn = KNN(k = sqrt(len(X_train)))
        
        if DETAILED_OUTPUT:
            print(f'\n\tTraining with gesture: \'{cur_gesture}\'...')
        
        fuzzy.train(X_train)
        svm.train(X_train)
        knn.train(X_train)
        
        if DETAILED_OUTPUT:
            print('\tTraining is finished!')
        
        cur_fuzzy_res = []
        cur_svm_res = []
        cur_knn_res = []
        
        if DETAILED_OUTPUT:
            print(f'\tStart testing for {len(X_test)} samples...')
            print('\r\t\t0%', end='')
        for record, i in zip(X_test, range(1, len(X_test) + 1)):
            cur_fuzzy_res.append(fuzzy.classify(record))
            cur_svm_res.append(svm.classify(record))
            cur_knn_res.append(knn.classify(record))
            
            if DETAILED_OUTPUT:
                print('\r\t\t{:.2f}%'.format(round(i / len(X_test) * 100, 2)), end='')
        if DETAILED_OUTPUT:
            print('\r\tTesting is finished')
        
        for classifier, res in zip(['FUZZY', 'SVM', 'KNN'], [cur_fuzzy_res, cur_svm_res, cur_knn_res]):
        #for classifier, res in zip(['FUZZY', 'SVM'], [cur_fuzzy_res, cur_svm_res]):
            results, answers = zip(*sorted(zip(res, y_test)))
            fpr, tpr, thresholds = roc_curve(answers, results)
            roc_auc = auc(fpr, tpr)
            
            if DETAILED_OUTPUT:
                print(f"\t\tArea under ROC curve for {classifier}: {roc_auc}")
            
            if classifier == 'FUZZY':
                fuzzy_res.append(roc_auc)
            elif classifier == 'SVM':
                svm_res.append(roc_auc)
            else:
                knn_res.append(roc_auc)
                


print('\n\nSENSORS:', SENSORS)
print('\nTRANSFORMS:', ', '.join([''.join(i) for i in [f.__name__ for f in TRANSFORMS]]))
print('\nKWARGS:', KWARGS)
print()
print('Fuzzy -', np.mean(np.array(fuzzy_res)))
print('SVM -', np.mean(np.array(svm_res)))
print('KNN -', np.mean(np.array(knn_res)))
print()
