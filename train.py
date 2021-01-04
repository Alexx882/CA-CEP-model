LAYER_NAME = 'CallTypeLayer'
import sys
if len(sys.argv) > 1:
    LAYER_NAME = sys.argv[1]

print(f"Working on {LAYER_NAME}")

##########

import json
from entities import Cluster
import collections
import numpy as np
from typing import Iterable

def get_evolution_label(old_size: int, new_size: int) -> int:
    '''Returns the evolution label as int by mapping 0..4 to {continuing, shrinking, growing, dissolving, forming}.'''
    if old_size == new_size:
        return 0 # continuing
    if old_size == 0 and new_size != 0:
        return 4 # forming
    if old_size != 0 and new_size == 0:
        return 3 # dissolving
    if old_size > new_size:
        return 1 # shrinking
    if old_size < new_size:
        return 2 # growing

def get_cyclic_time_feature(time: int, max_time_value: int = 52) -> (float, float):
    return (np.sin(2*np.pi*time/max_time_value),
            np.cos(2*np.pi*time/max_time_value))

def create_metrics_training_data(N: int = 3, layer_name: str = 'CallTypeLayer') -> Iterable:
    """
    A single metrics training data point should look like this:

    (cluster_size, cluster_std_dev, cluster_scarcity, cluster_import1, cluster_import2, time_info) ^ N, evolution_label
    time_info ... the time as 2d cyclic feature, i.e. time_info := (time_f1, time_f2)

    The first tuple represents metrics from the cluster in t_i-(N-1).
    The Nth tuple represents metrics from the cluster in t_i.
    The label is one of {continuing, shrinking, growing, dissolving, forming} \ {splitting, merging} and identifies the change for t_i+1.
    
    :param N: number of cluster metric tuples
    """
    
    path_in = f"input/metrics/{layer_name}.json"
    with open(path_in, 'r') as file:
        data = [Cluster.create_from_dict(cl_d) for cl_d in json.loads(file.read())]

    data.sort(key=lambda cl: (cl.cluster_id, cl.time_window_id))

    # manually prepare deque with N metric_tuples + evolution label
    tuples = []
    prev_cluster_id = -1

    for i, cur_cluster in enumerate(data[:-1]):

        if cur_cluster.cluster_id != data[i+1].cluster_id:
            # next cluster slice in list will be another cluster id -> restart deque and skip adding the current (last) cluster slice
            tuples = []
            continue

        cur_metrics = (cur_cluster.size, cur_cluster.std_dev, cur_cluster.scarcity, cur_cluster.importance1, cur_cluster.importance2, get_cyclic_time_feature(cur_cluster.get_time_info()))

        # deque function: adding N+1st element will remove oldest one
        if len(tuples) == N:
            tuples.pop(0)
        tuples.append(cur_metrics)

        label = get_evolution_label(cur_cluster.size, data[i+1].size)

        if len(tuples) == N:
            yield list(tuples) + [label]

###########

def flatten_metrics_datapoint(datapoint: list) -> ('X', 'Y'):
    '''
    Flattens a single metrics data point in the form:
    [(cluster_size, cluster_variance, cluster_density, cluster_import1, cluster_import2, (time_f1, time_f2))^N, evolution_label]
    to:
    (X: np.array, evolution_label)
    '''
    flat_list = []
    for entry in datapoint[:-1]: # for all x
        flat_list.extend(entry[:-1]) # add all number features except the time tuple
        flat_list.extend(entry[-1]) # add time tuple

    # flat_list.append(datapoint[-1]) # add y

    return np.asarray(flat_list), datapoint[-1]

##########

def convert_metrics_data_for_training(data: Iterable) -> ('nparray with Xs', 'nparray with Ys'):
    '''Flattens and splits metrics data to match ML conventions.'''
    X = []
    Y = []

    for element in data:
        x, y = flatten_metrics_datapoint(element)
        
        X.append(x)
        Y.append(y)

    return (np.asarray(X), np.asarray(Y))

##########

data = list(create_metrics_training_data(layer_name=LAYER_NAME))

import random
random.shuffle(data)

# test data size: 20%
test_size = int(len(data) * .2) 
train_metrics = data[:-test_size]
test_metrics = data[len(data)-test_size:]

print(f"Working with: {len(train_metrics)} training points + {len(test_metrics)} test points ({len(test_metrics)/(len(train_metrics)+len(test_metrics))}).")

X_train, Y_train = convert_metrics_data_for_training(train_metrics)
X_test, Y_test = convert_metrics_data_for_training(test_metrics)


import collections
import statistics as stat
print(f"Label Occurrences: Total = {collections.Counter(Y_train.tolist() + Y_test.tolist())}, Training = {collections.Counter(Y_train)}, Test = {collections.Counter(Y_test)}")
try:
    print(f"Label Majority Class: Training = {stat.mode(Y_train)}, Test = {stat.mode(Y_test)}\n")
except stat.StatisticsError:
    print(f"Label Majority Class: no unique mode; found 2 equally common values")

###########

# train
from sklearn import svm

svc = svm.SVC(kernel='linear')
svc.fit(X_train, Y_train)

# verify
import sklearn

pred_Y = svc.predict(X_test)

print(sklearn.metrics.classification_report(y_true=Y_test, y_pred=pred_Y))

# export
import pickle 

import os
if not os.path.exists('output'):
    os.makedirs('output')

with open(f'output/{LAYER_NAME}.model', 'wb') as file:
    b = pickle.dump(svc, file)
