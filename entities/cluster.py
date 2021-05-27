# from __future__ import annotations
from typing import Dict, List, Iterable, Any, Tuple
from entities.timewindow import TimeWindow
import numpy as np
import scipy
from processing import ClusterMetricsCalculatorFactory


class Cluster:
    '''A cluster from one time window containing all metrics used for machine learning.'''

    def __init__(self, time_window_id: Any, cluster_id: Any, cluster_nodes: List[dict], cluster_feature_names: List[str], nr_layer_nodes: int, layer_diversity: int, 
        global_cluster_center, global_center_distance=None):
        self.time_window_id = time_window_id
        self.cluster_id = cluster_id

        metrics_calculator = ClusterMetricsCalculatorFactory.create_metrics_calculator(cluster_nodes, cluster_feature_names, nr_layer_nodes, layer_diversity)

        self.size = metrics_calculator.get_size()
        self.std_dev = metrics_calculator.get_standard_deviation()
        self.scarcity = metrics_calculator.get_scarcity()
        
        self.importance1 = metrics_calculator.get_importance1()
        self.importance2 = metrics_calculator.get_importance2()
        
        self.range_ = metrics_calculator.get_range()
        self.center = metrics_calculator.get_center()

        self.global_center_distance = \
            scipy.spatial.distance.euclidean(self.center, global_cluster_center) \
            if self.size > 0 \
            else  0

    def get_time_info(self) -> int:
        '''Returns the week of the time tuple str, eg. 25 for "(2014, 25)".'''
        str_tuple = self.time_window_id
        return int(str_tuple.split(',')[1].strip()[:-1])

    def __repr__(self):
        return str(self.__dict__)

    def __str__(self):
        return f"Cluster({self.time_window_id}, {self.cluster_id}, " \
        f"{self.size}, {self.std_dev}, {self.scarcity}, " \
        f"{self.importance1}, {self.importance2}, " \
        f"{self.range_}, {self.center})"

    @staticmethod
    def create_multiple_from_time_window(time_window: TimeWindow, cluster_feature_names: List[str], global_cluster_centers: Dict[str, Tuple[float]]) -> Iterable['Cluster']:
        total_layer_nodes = sum([len(nodes) for nodes in time_window.clusters.values()])
        
        layer_diversity = len([nodes for nodes in time_window.clusters.values() if len(nodes) > 0])

        for cluster_nr, cluster_nodes in time_window.clusters.items():
            yield Cluster(time_window.time, cluster_nr, cluster_nodes, cluster_feature_names, total_layer_nodes, layer_diversity, global_cluster_centers[cluster_nr])

    @staticmethod
    def create_from_dict(dict_) -> 'Cluster':
        cl = Cluster(0, 0, [], 'None', 0, 0, None)
        cl.__dict__.update(dict_)
        return cl
