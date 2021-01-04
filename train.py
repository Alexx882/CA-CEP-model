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

import numpy as np
import pandas as pd
import collections
import statistics as stat

def balance_dataset(X: np.array, Y: np.array, imbalance_threshold=.3) -> ('X: np.array', 'Y: np.array'):
    '''Balances an unbalanced dataset by ignoring elements from the majority label, so that majority-label data size = median of other cluster sizes.'''
    y = Y.tolist()
    counter = collections.Counter(y)
    print(f"Label Occurrences: Total = {counter}")

    # find key with max values
    max_key = max(counter, key=lambda k: counter[k])
    max_val = counter[max_key]

    unbalanced_labels = all([v < max_val * (1-imbalance_threshold) for k, v in counter.items() if k != max_key]) 
    if unbalanced_labels: # if all other labels are >=30% less frequent than max_key
        median_rest = int(stat.median([v for k, v in counter.items() if k != max_key]))
        print(f"Labels are unbalanced, keeping {median_rest} for label {max_key}")
        
        # merge X and Y
        data = np.append(X, Y.reshape(Y.shape[0], 1), 1)
        df = pd.DataFrame(data, columns=['_']*21+['label'])

        # take only median_rest for the max_key label
        max_labeled_data = df.loc[df['label'] == max_key].sample(n=median_rest)
        other_labeled_data = df.loc[df['label'] != max_key]
        balanced_data = pd.concat([max_labeled_data, other_labeled_data])
        balanced_data = balanced_data.sample(frac=1) # shuffle

        X = balanced_data.loc[:, balanced_data.columns != 'label'].to_numpy()
        Y = balanced_data.loc[:, balanced_data.columns == 'label'].to_numpy()
        Y = Y.reshape(Y.shape[0],).astype(int)
       
    return X, Y

def get_training_data(layer_name='CallTypeLayer', test_dataset_frac=.2) -> '(X_train, Y_train, X_test, Y_test)':
    # load metrics data from disk
    data: Iterable = create_metrics_training_data(layer_name=layer_name)
    
    # convert to X and Y
    X, Y = convert_metrics_data_for_training(data)
    X, Y = balance_dataset(X, Y)
    
    # split in training and test set
    test_size = int(X.shape[0] * test_dataset_frac) 
    X_train = X[test_size:]
    Y_train = Y[test_size:]
    X_test = X[:test_size]
    Y_test = Y[:test_size]

    print(f"\nWorking with: {X_train.shape[0]} training points + {X_test.shape[0]} test points ({X_test.shape[0]/(X_train.shape[0]+X_test.shape[0])}).")
    print(f"Label Occurrences: Total = {collections.Counter(Y_train.tolist() + Y_test.tolist())}, "\
          f"Training = {collections.Counter(Y_train)}, Test = {collections.Counter(Y_test)}")
    try:
        print(f"Label Majority Class: Training = {stat.mode(Y_train)}, Test = {stat.mode(Y_test)}\n")
    except stat.StatisticsError:
        print(f"Label Majority Class: no unique mode; found 2 equally common values")

    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = get_training_data(LAYER_NAME)

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
