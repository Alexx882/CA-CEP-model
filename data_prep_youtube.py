
########################
#### Single-Context ####
########################

from typing import List
import json
import os
from entities import TimeWindow, Cluster

def store_metrics_for_clusters(layer_name: str, feature_names: List[str]):
    '''
    :param layer_name: Name of the layer for which multiple time windows exist
    :param feature_names: Features of the layer
    '''
    print(f"Working on {layer_name}")

    path_in = f'_predictions/timeslices/{layer_name}'
    path_out = f'_predictions/metrics/{layer_name}.json'

    complete_clusters: List[Cluster] = []

    for root, _, files in os.walk(path_in):
        for f in files:
            with open(os.path.join(root, f), 'r') as file:
                # for each time window json
                json_slice = json.loads(file.read())
                time_window = TimeWindow.create_from_serializable_dict(json_slice)

                # create all clusters + metrics for one time window
                clusters = Cluster.create_multiple_from_time_window(time_window, feature_names)
                complete_clusters.extend(clusters)
        
    # store the cluster metrics
    with open(path_out, 'w') as file:
        file.write(json.dumps([cl.__dict__ for cl in complete_clusters]))


layers = [
    ['CategoryLayer', 'category_id'],
    ['ViewsLayer', 'views'],
    ['LikesLayer', 'likes'],
    ['DislikesLayer', 'dislikes'],
    ['CommentCountLayer', 'comment_count'],
    ['CountryLayer', 'country_id'],
    ['TrendDelayLayer', 'trend_delay'],
]

for layer in layers:
    store_metrics_for_clusters(layer[0], layer[1])


COLUMNS = ['cluster_size', 'cluster_variance', 'cluster_density', 'cluster_import1', 'cluster_import2', 
        'cluster_area', 'cluster_center_x', 'cluster_center_y', 'time_f1', 'time_f2']*3 + ['evolution_label']


import json
from entities import Cluster
import collections
import numpy as np
from typing import Iterable

def get_evolution_label(old_size: int, new_size: int) -> int:
    '''Returns the evolution label as int by mapping 0..4 to {continuing, shrinking, growing, dissolving, forming}.'''
    if old_size == new_size:
        return 0 # continuing
    if old_size == 0 and new_size > 0:
        return 4 # forming
    if old_size > 0 and new_size == 0:
        return 3 # dissolving
    if old_size > new_size:
        return 1 # shrinking
    if old_size < new_size:
        return 2 # growing

def get_cyclic_time_feature(time: int, max_time_value: int = 52) -> (float, float):
    return (np.sin(2*np.pi*time/max_time_value),
            np.cos(2*np.pi*time/max_time_value))

def create_metrics_training_data(layer_name: str, N: int = 3) -> Iterable[list]:
    """
    Loads the metrics training data for an individual layer from disk.
    A single metrics training data point should look like this:

    (cluster_size, cluster_std_dev, cluster_scarcity, cluster_import1, cluster_import2, cluster_range, cluster_center_x, cluster_center_y, time_info) ^ N, evolution_label
    time_info ... the time as 2d cyclic feature, i.e. time_info := (time_f1, time_f2)

    The first tuple represents metrics from the cluster in t_i-(N-1).
    The Nth tuple represents metrics from the cluster in t_i.
    The label is one of {continuing, shrinking, growing, dissolving, forming} \ {splitting, merging} and identifies the change for t_i+1.
    
    :param N: number of cluster metric tuples
    :param layer_name: the name of the layer metrics json file
    """
    
    path_in = f"_predictions/metrics/{layer_name}.json"
    with open(path_in, 'r') as file:
        data = [Cluster.create_from_dict(cl_d) for cl_d in json.loads(file.read())]

    data.sort(key=lambda cl: (cl.cluster_id, cl.time_window_id))

    # manually prepare deque with N metric_tuples + evolution label
    tuples = []

    for i, cur_cluster in enumerate(data[:-1]):

        if cur_cluster.cluster_id != data[i+1].cluster_id:
            # next cluster slice in list will be another cluster id -> restart deque and skip adding the current (last) cluster slice
            tuples = []
            continue

        cur_metrics = (cur_cluster.size, cur_cluster.std_dev, cur_cluster.scarcity, cur_cluster.importance1, cur_cluster.importance2, cur_cluster.range_, cur_cluster.center[0], cur_cluster.center[1], get_cyclic_time_feature(cur_cluster.get_time_info()))

        # deque function: adding N+1st element will remove oldest one
        if len(tuples) == N:
            tuples.pop(0)
        tuples.append(cur_metrics)

        if len(tuples) == N:
            label = get_evolution_label(cur_cluster.size, data[i+1].size)
            yield list(tuples) + [label]


def flatten_metrics_datapoint(datapoint: list) -> ('X, y: np.array'):
    '''
    Flattens a single metrics data point in the form:
    [(cluster_size, cluster_variance, cluster_density, cluster_import1, cluster_import2, cluster_range, cluster_center, (time_f1, time_f2))^N, evolution_label]
    to:
    (X, y: np.array
    '''
    flat_list = []
    for entry in datapoint[:-1]: # for all x
        flat_list.extend(entry[:-1]) # add all number features except the time tuple
        flat_list.extend(entry[-1]) # add time tuple

    flat_list.append(datapoint[-1]) # y
    return np.asarray(flat_list)


import pandas as pd

def convert_metrics_data_to_dataframe(data: Iterable, columns: list, flattening_method: 'callable') -> pd.DataFrame:
    '''Flattens and splits metrics data to match ML conventions.'''
    training_data = []

    for element in data:
        xy: 'np.array' = flattening_method(element)
        
        training_data.append(xy)

    return pd.DataFrame(data=training_data, columns=columns)


import numpy as np
import pandas as pd
from pandas import DataFrame
import collections
import statistics as stat

def balance_dataset(df: DataFrame) -> DataFrame:
    # TODO
    return df

def store_training_data(layer_name: str):
    # load metrics data from disk
    data: Iterable = create_metrics_training_data(layer_name=layer_name)
    
    # flatten and convert to df
    df = convert_metrics_data_to_dataframe(data, columns=COLUMNS, flattening_method=flatten_metrics_datapoint)

    # balance df
    df = balance_dataset(df)

    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    df.to_csv(f'_predictions/cluster_metrics/data/{layer_name}.csv')


for name, _ in layers:
    store_training_data(layer_name=name)


#######################
#### Cross-Context ####
#######################

from typing import List, Tuple
import statistics as stat
import json
import os
from entities import TimeWindow, Layer
from processing import ClusterMetricsCalculatorFactory

def calculate_center(cluster: dict, features: list) -> Tuple[float]:
    calc = ClusterMetricsCalculatorFactory.create_metrics_calculator(cluster['nodes'], features, 1, 1)
    return calc.get_center()
    # if '--' in label:
    #     return (stat.mean([float(e) for e in label.split('--')]), 0)
    # else:
    #     return [float(e) for e in label.replace('(', '').replace(')', '').split(',')]

def store_metrics_for_layers(layer_name: str = 'CallTypeLayer', feature_names: List[str] = ['call_type']):
    print(f"Working on {layer_name}")

    # load global cluster centers
    path_in = f'_predictions/clusters/{layer_name}.json'
    with open(path_in, 'r') as file:
        clusters = json.loads(file.read())
        cluster_centers: Dict[str, Tuple[float]] = { 
            str(cluster['cluster_label']): calculate_center(cluster, feature_names) 
            for cluster in clusters 
            if cluster['label'] != 'noise' 
            }

    # load time windows 
    all_layers: List[Layer] = []
    path_in = f'_predictions/timeslices/{layer_name}'
    for root, _, files in os.walk(path_in):
        for f in files:
            with open(os.path.join(root, f), 'r') as file:
                json_time_slice = json.loads(file.read())
                time_window = TimeWindow.create_from_serializable_dict(json_time_slice)

                layer = Layer.create_from_time_window(time_window, feature_names, cluster_centers)
                all_layers.append(layer)
        
    # store the layer metrics
    path_out = f'_predictions/layer_metrics/{layer_name}.json'
    with open(path_out, 'w') as file:
        file.write(json.dumps([l.__dict__ for l in all_layers]))


for layer in layers:
    store_metrics_for_layers(layer[0], layer[1])


COLUMNS = ['n_nodes', 'n_clusters', 'entropy', 
           'relative_cluster_sizes', 'cluster_centers', 'distance_from_global_centers', 
           'time_f1', 'time_f2'] * 2 + ['evolution_label']


def get_cyclic_time_feature_from_time_window(time: str) -> (float, float):
    return get_cyclic_time_feature(int(time.replace('(', '').replace(')', '').split(',')[1]))


from typing import Iterable, List, Dict, Any
import json
from entities import Layer, Cluster

def get_layer_metrics(layer: Layer):
    return (layer.n_nodes, layer.n_clusters, layer.entropy,
     layer.relative_cluster_sizes, layer.centers, layer.distances_from_global_centers,
     get_cyclic_time_feature_from_time_window(layer.time_window_id))

def create_layer_metrics_training_data(layer_name: str, reference_layer: str, N: int = 2) -> Iterable:
    """
    Loads the metrics training data for an individual layer from disk.
    A single metrics training data point should look like this:

    [(n_nodes, n_clusters, entropy,
     (relative_cluster_size)^M, (cluster_centers)^M, (distance_from_global_centers)^M, 
     (time1, time2))^N, 
     cluster_number, evolution_label]

    The first tuple represents metrics from the reference layer in t_i-(N-1).
    The Nth tuple represents metrics from the reference layer in t_i.
    The reference_layer has M clusters in total, this might differ from the number of clusters in layer_name.
    The cluster number identifies the cluster for which the evolution_label holds. 
    The label is one of {continuing, shrinking, growing, dissolving, forming} \ {splitting, merging} and identifies the change for a cluster in the layer layer_name for t_i.
    
    # TODO N is not implemented and fixed to 2
    """
    
    with open(f'_predictions/metrics/{layer_name}.json') as file:
        cluster_metrics: List[Cluster] = [Cluster.create_from_dict(e) for e in json.loads(file.read())]
        cluster_ids = {c.cluster_id for c in cluster_metrics}
        cluster_metrics: Dict[Any, Cluster] = {(c.time_window_id, c.cluster_id): c for c in cluster_metrics}
        
    with open(f'_predictions/layer_metrics/{reference_layer}.json') as file:
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

        current_layer_metric_tuple = get_layer_metrics(current_layer_metric)
        prev_layer_metric_tuple = get_layer_metrics(prev_layer_metric)

        # ...then load the current and previous cluster metrics for all clusters in the layer_name
        for cluster_id in cluster_ids:
            current_cluster_metric = cluster_metrics[(current_time_key, cluster_id)]
            prev_cluster_metric = cluster_metrics[(prev_time_key, cluster_id)]
            evolution_label = get_evolution_label(prev_cluster_metric.size, current_cluster_metric.size)

            # yield each combination of reference layer metrics to clusters
            yield [prev_layer_metric_tuple, current_layer_metric_tuple, int(cluster_id), evolution_label]

        prev_time_key = current_time_key

    
def flatten_layer_metrics_datapoint(datapoint: list) -> ('X, y: np.array'):
    '''
    Flattens a single layer metrics data point in the form:
    [(n_nodes, n_clusters, entropy,
     (relative_cluster_size)^M, (cluster_centers)^M, (distance_from_global_centers)^M, 
     (time1, time2))^N, 
     cluster_number, evolution_label]
    to:
    (X, y: np.array)
    '''
    flat_list = []
    for layer_metric_tuple in datapoint[:-2]: # for all x
        flat_list.append(layer_metric_tuple[0:2]) # n_nodes, n_clusters, entropy
        flat_list.extend(layer_metric_tuple[3]) # rel sizes
        flat_list.extend(layer_metric_tuple[4]) # centers
        flat_list.extend(layer_metric_tuple[5]) # distances
        flat_list.extend(layer_metric_tuple[6]) # time1/2

    flat_list.append(datapoint[-2]) # cluster num
    flat_list.append(datapoint[-1]) # y

    return np.asarray(flat_list)


import numpy as np
import pandas as pd
from pandas import DataFrame
import collections
import statistics as stat

def balance_dataset(df: DataFrame) -> ('X: np.array', 'Y: np.array'):
    '''Balances an unbalanced dataset by ignoring elements from the majority label, so that majority-label data size = median of other cluster sizes.'''
    return df

def store_training_data(layer_name='CallTypeLayer', reference_layer_name='CallTypeLayer') -> '(X_train, Y_train, X_test, Y_test)':
    # load metrics data from disk
    data: Iterable = create_layer_metrics_training_data(layer_name=layer_name, reference_layer=reference_layer_name)
    
    # convert to X and Y
    df = convert_metrics_data_to_dataframe(data, columns=COLUMNS, flattening_method=flatten_layer_metrics_datapoint)
    
    # balance df
    df = balance_dataset(df)
    
    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    df.to_csv(f'_predictions/layer_metrics/data/{layer_name}.csv')


store_training_data(layer_name='CategoryLayer', reference_layer_name='CountryLayer')
