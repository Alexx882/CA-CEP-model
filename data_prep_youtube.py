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

def convert_metrics_data_to_dataframe(data: Iterable, columns) -> pd.DataFrame:
    '''Flattens and splits metrics data to match ML conventions.'''
    training_data = []

    for element in data:
        xy: 'np.array' = flatten_metrics_datapoint(element)
        
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
    df = convert_metrics_data_to_dataframe(data, columns=COLUMNS)

    # balance df
    df = balance_dataset(df)

    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    df.to_csv(f'_predictions/cluster_metrics/data/{layer_name}.csv')


for name, _ in layers:
    store_training_data(layer_name=name)