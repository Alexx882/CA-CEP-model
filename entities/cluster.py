from __future__ import annotations
from typing import Dict, List, Iterable, Any
from entities.timewindow import TimeWindow
import numpy as np

class Cluster:
    '''A cluster from one time window containing all metrics used for machine learning.'''

    def __init__(self, time_window_id: Any, cluster_id: Any, cluster_nodes: List[dict], cluster_feature: str, nr_layer_nodes: int, diversity: int):
        self.time_window_id = time_window_id
        self.cluster_id = cluster_id

        self.size = len(cluster_nodes)
        feature_values = [node[cluster_feature] for node in cluster_nodes]
        self.variance = np.var(feature_values) if len(feature_values) > 0 else 0
        self.density = self._calculate_density(feature_values)
        
        self.importance1 = float(len(cluster_nodes)) / nr_layer_nodes if len(cluster_nodes) > 0 else 0
        self.importance2 = 1.0 / diversity if len(cluster_nodes) > 0 else 0

    def _calculate_density(self, feature_values):
        '''Returns the density as cluster_range / # cluster_nodes, or 0 if len(nodes)=0.'''
        if len(feature_values) == 0:
            return 0

        range_ = max(feature_values) - min(feature_values)
        return float(range_) / len(feature_values)

    def get_time_info(self) -> int:
        '''Returns the week of the time tuple str, eg. 25 for "(2014, 25)".'''
        str_tuple = self.time_window_id
        return int(str_tuple.split(',')[1].strip()[:-1])

    def __repr__(self):
        return str(self.__dict__)

    def __str__(self):
        return f"Cluster({self.cluster_id}, " \
        f"{self.size}, {self.variance}, {self.density}, " \
        f"{self.importance1}, {self.importance2})"

    @staticmethod
    def create_from_time_window(time_window: TimeWindow, cluster_feature: str) -> Iterable[Cluster]:
        total_layer_nodes = sum([len(nodes) for nodes in time_window.clusters.values()])
        
        diversity = len([nodes for nodes in time_window.clusters.values() if len(nodes) > 0])

        for cluster_nr, cluster_nodes in time_window.clusters.items():
            yield Cluster(time_window.time, cluster_nr, cluster_nodes, cluster_feature, total_layer_nodes, diversity)

    @staticmethod
    def create_from_dict(dict_):
        cl = Cluster(0, 0, [], 0, 0, 0)
        cl.__dict__.update(dict_)
        return cl
