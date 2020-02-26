#!/usr/bin/env python3
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl

import seaborn #plotting lib, but just adding makes the matplotlob plots better

from pprz_data.pprz_data import DATA
import numpy as np

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import classification_report

import sklearn.model_selection as ms


def plot_all(data):
    mpl.style.use('seaborn')
    # fig=plt.figure(figsize=(19,7))
    # df_labelled.plot(y=['m1', 'alt'], figsize=(17,7));plt.show()
    data.plot(subplots=True, figsize=(12,10));plt.show()

def add_time_history(X,y,n_step=3):
    time_len= X.shape[0] #-n_step
    column_len = X.shape[1]

    xx=np.zeros((time_len,column_len*n_step))

    for i in range(n_step, time_len):
        for j in range(n_step):
            xx[i,j*column_len:(j+1)*column_len] = X[i-j]
            
    xx = xx[n_step:X.shape[0],:]
    yy = y[n_step:]
    
    return xx,yy


def main():
    ac_id = '9'
    filename = '../data/jumper_2nd.data'
    
    data = DATA(filename, ac_id, data_type='fault')
    # Labelled data can be built from the data class directly
    df_labelled = data.get_labelled_data()
    df_labelled.plot(subplots=True, figsize=(17,25));plt.show()
    # df_labelled.describe()
    labeled_data = df_labelled[300:500]
    df_flight = labeled_data.copy()
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
    st=395; fn=450
    flight = df_flight[st:fn].copy()

    # flight['Az'] = flight['Az']+9.81
    X_pre = flight[columns].values # Features
    y_pre = flight.fault.values   # Labels
    # n_step = 3 # time history step that we would like to add
    X,y = add_time_history(X_pre,y_pre,n_step=10)

    # # Define the scaler and Scale it
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    # # y = scaler.fit_transform(y)

    # Train, Test, Validation Sets Splits

    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = ms.train_test_split(X_train, y_train, test_size=0.2, random_state=42)


    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)

    clf = SVC(kernel='rbf', gamma='auto', C=35).fit(X_train_transformed, y_train)
    # clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
    cv = ShuffleSplit(n_splits=5, test_size=0.3 , random_state=42)

    # scores = cross_val_score(clf, X, y, scoring='accuracy', cv=5) # default is accuracy
    # print(scores)
    # print("Accuracy: %0.2f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

    scores = cross_val_score(clf, X_test_transformed, y_test, scoring='accuracy', cv=cv) # default is accuracy, f1_macro, roc_auc_score
    print(scores)
    print("Accuracy: %0.2f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

    scores = cross_val_score(clf, X_test_transformed, y_test, scoring='f1_macro', cv=cv)
    print(scores)
    print("F1-Score: %0.2f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

    X_transformed = scaler.transform(X)
    scores = cross_val_score(clf, X_transformed, y, scoring='f1_macro', cv=cv)
    print(scores)
    print("F1-Score: %0.2f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))


    X_val_transformed = scaler.transform(X_val)
    y_pred = clf.predict(X_val_transformed)
    target_names = ['nominal', 'faulty']
    print(classification_report(y_val, y_pred, target_names=target_names))

    print('Finished !!!')
    
    plot_all(data.df_All)


if __name__ == "__main__":
    main()