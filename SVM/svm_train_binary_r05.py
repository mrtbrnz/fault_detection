#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sklearn.model_selection as ms
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from pprz_data.pprz_data import DATA 

import joblib

# Elgiz
def add_time_history_0(X,steps_added):
    time_len = X.shape[0]
    num_origin_features = X.shape[1]
    num_steps = steps_added + 1
    X_added_feat = np.zeros((time_len, num_origin_features * num_steps))
    v_new = np.zeros((X.shape[0],num_steps))

    for i in range(X.shape[1]):
        v_a = X[:,i]
        v_new[:,-1] = v_a

        for j in range(1,num_steps):
            v_a[1:len(v_a)] = v_a[0:-1]
            v_new[:,num_steps - 1 - j] = v_a
            X_added_feat[:,(i * num_steps): ((i+1) * num_steps)]= v_new
    return X_added_feat

# Adds measurements of previous n_step - 1 time steps. 
def add_time_history_1(X,y,n_step=3):
   time_len = X.shape[0]
   column_len = X.shape[1]

   xx = np.zeros((time_len,column_len * n_step))

   for i in range(n_step,time_len):
       for j in range(n_step):
           xx[i,j*column_len:(j+1)*column_len] = X[i-j]

   xx = xx[n_step:X.shape[0],:]
   yy = y[n_step:]

   return xx,yy

# Adds measurements of previous n_step - 1 time steps. 
def add_time_history_2(X,y,n_step=3):
   time_len = X.shape[0]
   column_len = X.shape[1]

   xx = np.zeros((time_len,column_len * n_step))

   for i in range(n_step,time_len):
       for j in range(column_len):
           xx[i,j*n_step:(j+1)*n_step] = X[(i+1-n_step):(i+1),j]

   xx = xx[n_step:X.shape[0],:]
   yy = y[n_step:]

   return xx,yy

# Selection of flight data file and aircraft id (ac_id) - required from the user
# select the aircraft of interest in the filename.data

def main():
    # ac_id = '9'
    # filename = '../data/jumper_2nd.data'

    ac_id = '20'
    filename = '../data/20_07_03__11_13_15_SD.data'

    # taking only the data variables of interest defined in the DATA class by passing 'fault' to data_type 
    data = DATA(filename, ac_id, data_type='fault')
    df_labelled = data.get_labelled_data()

    df_flight = df_labelled[1500:1830].copy()
    # df_flight.describe()
    df_flight = df_flight.assign(fault = 0)                                                      
    cond1 = (df_flight['add1'] > 0.005) | (df_flight['add1'] < -0.005)
    cond2 = (df_flight['add2'] > 0.005) | (df_flight['add2'] < -0.005) 
    cond3 = (df_flight['m1'] < 1.0) | (df_flight['m2'] < 1.0)        
    cond = cond1 | cond2 | cond3
    df_flight.loc[cond, 'fault'] = 1 
    cond4 = (df_flight['mode'] == 2.0)                                    
    # cond2 = (df_flight['mode'] == 1.0)                                  
    cond = cond4 # | cond2                                                
    df_flight = df_flight[cond]                                                          
    print('Create the Feature and Label List')                                          
    columns=['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']                                        
    # TO BE DONE : selection of flight interval
    # Human selection of time interval of interest
    st=1500; fn=1830                                                                      
    flight = df_flight[st:fn].copy()                                                             
    # flight['Az'] = flight['Az']+9.81                                                           
    X_pre = flight[columns].values # Features          
    y = flight.fault.values   # Labels
  #  print(X_pre.shape)
    # X = X_pre
    # X = add_time_history_0(X_pre,steps_added = 3) 
    X,y = add_time_history_2(X_pre,y,n_step= 3)
   # print(X_feat_added.shape)

    # Train, Test, Validation Sets Splits

    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # stratify : If not None, data is split in a stratified fashion, using this as the class labels.
   
    # If some outliers are present in the set, robust scalers or transformers are more appropriate.  
    # Define the scaler and Scale it
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test) 

    # Set the parameters by cross-validation
    #tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
    #                 'C': [1, 10, 100, 1000]},
    #                {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]}]
    score = 'f1'

    print("# Tuning hyper-parameters for %s" % score)
    print()

    # cv : For integer/None inputs, if the estimator is a classifier and y is either binary or multiclass, StratifiedKFold is used. In all other cases, KFold is used.
    clf = GridSearchCV(SVC(), tuned_parameters, scoring='%s_micro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

    joblib.dump(clf, 'svm_model_binary_r05.joblib')
    joblib.dump(scaler, 'svm_scaler_binary_r05.joblib')

if __name__ == "__main__":
    import timeit
    # print(timeit.timeit("main()", setup="from __main__ import main", number=5))
    main()
