
MAX_DEPTHS: int = [5, 10, 15]
LAYER_NAME: str = 'CallTypeLayer'
REFERENCE_LAYER_NAME: str = 'DayTypeLayer'

import sys
if len(sys.argv) > 1:
    LAYER_NAME = sys.argv[1]
    REFERENCE_LAYER_NAME = sys.argv[2]

print(f"Working with params:")
from icecream import ic
ic(LAYER_NAME)
ic(REFERENCE_LAYER_NAME)

#######################


# TODO remove dup code

import numpy as np

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

def get_cyclic_time_feature_from_time_window(time: str) -> (float, float):
    return get_cyclic_time_feature(int(time.replace('(', '').replace(')', '').split(',')[1]))

#########

from typing import Iterable, List, Dict, Any
import json
from entities import Layer, Cluster

def create_metrics_training_data(N: int = 2, layer_name: str = 'CallTypeLayer', reference_layer: str = 'CallTypeLayer') -> Iterable:
    """
    Loads the metrics training data for an individual layer from disk.
    A single metrics training data point should look like this:

    [((relative_cluster_size) ^ M, entropy, (distance_from_global_center) ^ M, (time1, time2)) ^ N, cluster_number, evolution_label]

    The first tuple represents metrics from the reference layer in t_i-(N-1).
    The Nth tuple represents metrics from the reference layer in t_i.
    The reference_layer has M clusters in total, this might differ from the number of clusters in layer_name.
    The cluster number identifies the cluster for which the evolution_label holds. 
    The label is one of {continuing, shrinking, growing, dissolving, forming} \ {splitting, merging} and identifies the change for a cluster in the layer layer_name for t_i.
    
    # TODO what exactly should the classifier predict? 
    # all cluster changes, this would mean that cluster information has to be provided 

    # TODO N is not implemented and fixed to 2
    """
    
    with open(f'input/metrics/{layer_name}.json') as file:
        cluster_metrics: List[Cluster] = [Cluster.create_from_dict(e) for e in json.loads(file.read())]
        cluster_ids = {c.cluster_id for c in cluster_metrics}
        cluster_metrics: Dict[Any, Cluster] = {(c.time_window_id, c.cluster_id): c for c in cluster_metrics}
        
    with open(f'input/layer_metrics/{reference_layer}.json') as file:
        layer_metrics: List[Layer] = [Layer.create_from_dict(e) for e in json.loads(file.read())]
        layer_metrics: Dict[Any, Layer] = {l.time_window_id: l for l in layer_metrics}

    # load the time keys chronologically
    ordered_time_keys = list(layer_metrics.keys())
    ordered_time_keys.sort(key=lambda x: [int(v) for v in x.replace('(', '').replace(')', '').split(',')])
    
    # go through all time windows once...
    prev_time_key = ordered_time_keys[0]
    for current_time_key in ordered_time_keys[1:]:
        # ...and load the current and previous layer metrics in the reference_layer
        current_layer_metric = layer_metrics[current_time_key]
        prev_layer_metric = layer_metrics[prev_time_key]
        current_layer_metric_tuple = (current_layer_metric.relative_cluster_sizes, current_layer_metric.entropy, current_layer_metric.distances_from_global_centers, get_cyclic_time_feature_from_time_window(current_layer_metric.time_window_id))
        prev_layer_metric_tuple = (prev_layer_metric.relative_cluster_sizes, prev_layer_metric.entropy, prev_layer_metric.distances_from_global_centers, get_cyclic_time_feature_from_time_window(prev_layer_metric.time_window_id))

        # ...then load the current and previous cluster metrics for all clusters in the layer_name
        for cluster_id in cluster_ids:
            current_cluster_metric = cluster_metrics[(current_time_key, cluster_id)]
            prev_cluster_metric = cluster_metrics[(prev_time_key, cluster_id)]
            evolution_label = get_evolution_label(prev_cluster_metric.size, current_cluster_metric.size)

            # yield each combination of reference layer metrics to clusters
            yield [prev_layer_metric_tuple, current_layer_metric_tuple, int(cluster_id), evolution_label]

        prev_time_key = current_time_key

########

def flatten_metrics_datapoint(datapoint: list) -> ('X', 'Y'):
    '''
    Flattens a single layer metrics data point in the form:
    [((relative_cluster_size) ^ M, entropy, (distance_from_global_center) ^ M, (time1, time2)) ^ N, cluster_number, evolution_label]
    to:
    (X: np.array, evolution_label)
    '''
    flat_list = []
    for layer_metric_tuple in datapoint[:-2]:
        flat_list.extend(layer_metric_tuple[0]) # sizes
        flat_list.append(layer_metric_tuple[1]) # entropy
        flat_list.extend(layer_metric_tuple[2]) # distances
        flat_list.extend(layer_metric_tuple[3]) # time1/2

    flat_list.append(datapoint[-2]) # cluster num

    return np.asarray(flat_list), datapoint[-1]

#########

# TODO remove dup code
def convert_metrics_data_for_training(data: Iterable) -> ('nparray with Xs', 'nparray with Ys'):
    '''Flattens and splits metrics data to match ML conventions.'''
    X = []
    Y = []

    for element in data:
        x, y = flatten_metrics_datapoint(element)
        
        X.append(x)
        Y.append(y)

    return (np.asarray(X), np.asarray(Y))

###########

# TODO remove dup code

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
        df = pd.DataFrame(data, columns=['_']*X.shape[1]+['label'])

        # take only median_rest for the max_key label
        max_labeled_data = df.loc[df['label'] == max_key].sample(n=median_rest)
        other_labeled_data = df.loc[df['label'] != max_key]
        balanced_data = pd.concat([max_labeled_data, other_labeled_data])
        balanced_data = balanced_data.sample(frac=1) # shuffle

        X = balanced_data.loc[:, balanced_data.columns != 'label'].to_numpy()
        Y = balanced_data.loc[:, balanced_data.columns == 'label'].to_numpy()
        Y = Y.reshape(Y.shape[0],).astype(int)
       
    return X, Y

def get_training_data(layer_name='CallTypeLayer', reference_layer_name='CallTypeLayer', test_dataset_frac=.2) -> '(X_train, Y_train, X_test, Y_test)':
    # load metrics data from disk
    data: Iterable = create_metrics_training_data(layer_name=layer_name, reference_layer=reference_layer_name)
    
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

#########

X_train, Y_train, X_test, Y_test = get_training_data(layer_name=LAYER_NAME, reference_layer_name=REFERENCE_LAYER_NAME)

#########

for depth in MAX_DEPTHS:
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(max_depth=depth)
    classifier.fit(X_train, Y_train)

    # export
    import pickle 

    with open(f'output/layer_metrics/{depth}/{LAYER_NAME}_{REFERENCE_LAYER_NAME}.model', 'wb') as file:
        b = pickle.dump(classifier, file)


    # verify
    import sklearn

    pred_Y = classifier.predict(X_test)

    print(sklearn.metrics.classification_report(y_true=Y_test, y_pred=pred_Y))