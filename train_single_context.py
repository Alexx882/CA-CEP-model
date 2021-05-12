use_case = 'youtube'
layer_name = 'LikesLayer' 
DATA_SAMPLING_SIZE = 5000

import pandas as pd
from pandas import DataFrame

df: DataFrame = pd.read_csv(f'data/{use_case}/ml_input/single_context/{layer_name}.csv', index_col=0)

import numpy as np
import collections

def split_data(dataframe, test_dataset_frac=.2, shuffle=False) -> '(training_data, test_data)':
    if shuffle:
        dataframe = dataframe.sample(frac=1).reset_index(drop=True)

    training_size = int(len(dataframe) * (1-test_dataset_frac))

    train = dataframe[:training_size]
    test = dataframe[training_size:]

    y_train = train[train.columns[-1]]
    y_test = test[test.columns[-1]]
  
    print(f"\nWorking with: {len(train)} training points + {len(test)} test points ({len(test)/(len(test)+len(train))} test ratio).")
    print(f"Label Occurrences: Total = {collections.Counter(y_train.tolist() + y_test.tolist())}, "\
          f"Training = {collections.Counter(y_train)}, Test = {collections.Counter(y_test)}")
    # try:
    #     print(f"Label Majority Class: Training = {stat.mode(Y_train)}, Test = {stat.mode(Y_test)}\n")
    # except stat.StatisticsError:
    #     print(f"Label Majority Class: no unique mode; found 2 equally common values")

    return train, test

training, testing = split_data(df, shuffle=False)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

train_X = scaler.fit_transform(training)[:,:-1] # all except y
train_Y = training[training.columns[-1]]

test_X = scaler.transform(testing)[:,:-1] # all except y
test_Y = testing[testing.columns[-1]]

from processing import DataSampler

sampler = DataSampler()
train_X, train_Y = sampler.sample_fixed_size(train_X, train_Y, size=DATA_SAMPLING_SIZE)

from sklearn.decomposition import PCA

pca = PCA(n_components=8)

train_Xp = pca.fit_transform(train_X)
test_Xp = pca.transform(test_X)

import sklearn.metrics

def print_report(clfs: list, test_Xs: list, test_Y: 'y', titles: list):
    """
    Prints all reports.
    :param clfs: list of classifiers to evaluate
    :param test_Xs: list of test_X for the corresponding classifier at idx
    :param test_Y: true classes
    :param titles: list of titles for the classifiers at idx
    """
    for clf, test_X, title in zip(clfs, test_Xs, titles):
        pred_Y = clf.predict(test_X)        
        print(f"### {title} ###\n", sklearn.metrics.classification_report(y_true=test_Y, y_pred=pred_Y))

import pickle 
def export_model(model, model_name):
    with open(f'data/{use_case}/ml_output/single_context/{layer_name}_{model_name}.model', 'wb') as f:
        pickle.dump(model, f)

# NB
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(train_X, train_Y)

clf_p = GaussianNB()
clf_p.fit(train_Xp, train_Y)

print_report([clf, clf_p], [test_X, test_Xp], test_Y, ["X", "Xp"])

export_model(clf, 'nb_x')
export_model(clf_p, 'nb_xp')

# SVC
from sklearn.svm import SVC
c= 1

svc = SVC(kernel='linear', C=c)
svc.fit(train_X, train_Y)

svc_p = SVC(kernel='linear', C=c)
svc_p.fit(train_Xp, train_Y)

print_report([svc, svc_p], [test_X, test_Xp], test_Y, ["X", "Xp"])

export_model(svc, 'svc_x')
export_model(svc_p, 'svc_xp')

# KNN
from sklearn.neighbors import KNeighborsClassifier
n_neighbors=4
weights= 'uniform'
algo=  'auto'
leaf_size = 30

knnc = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algo, leaf_size=leaf_size)
knnc.fit(train_X, train_Y)

knnc_p = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights,  algorithm=algo, leaf_size=leaf_size)
knnc_p.fit(train_Xp, train_Y)

print_report([knnc, knnc_p], [test_X, test_Xp], test_Y, ["X", "Xp"])

export_model(knnc, 'knn_x')
export_model(knnc_p, 'knn_xp')

# DT
from sklearn.tree import DecisionTreeClassifier 
criterion = 'gini'
splitter = 'random'
max_depth = 10
min_samples_leaf = 1
min_impurity_decrease= 0 # impurity improvement needed to split
ccp_alpha = 0.001

dtc = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_impurity_decrease=min_impurity_decrease, ccp_alpha=ccp_alpha)
dtc.fit(train_X, train_Y)

dtc_p = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_impurity_decrease=min_impurity_decrease, ccp_alpha=ccp_alpha)
dtc_p.fit(train_Xp, train_Y)

print_report([dtc, dtc_p], [test_X, test_Xp], test_Y, ["X", "Xp"])

export_model(dtc, 'dt_x')
export_model(dtc_p, 'dt_xp')

# RF
from sklearn.ensemble import RandomForestClassifier
n_estimators = 100
criterion = 'entropy'
max_depth = None
min_samples_leaf = 2
min_impurity_decrease= 0
bootstrap=True

rfc = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap)
rfc.fit(train_X, train_Y)

rfc_p = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap)
rfc_p.fit(train_Xp, train_Y)

print_report([rfc, rfc_p], [test_X, test_Xp], test_Y, ["X", "Xp"])

export_model(rfc, 'rf_x')
export_model(rfc_p, 'rf_xp')

# Boosting
from sklearn.ensemble import AdaBoostClassifier
base_estimator= SVC(kernel='linear')
n_estimators= 100
algo = 'SAMME'
learning_rate = .3

bc = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators, algorithm=algo, learning_rate=learning_rate)
bc.fit(train_X, train_Y)

bc_p = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators, algorithm=algo, learning_rate=learning_rate)
bc_p.fit(train_Xp, train_Y)

print_report([bc, bc_p], [test_X, test_Xp], test_Y, ["X", "Xp"])

export_model(bc, 'boost_x')
export_model(bc_p, 'boost_xp')