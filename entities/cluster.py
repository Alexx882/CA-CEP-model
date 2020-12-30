from __future__ import annotations
from typing import Dict, List, Iterable, Any
from entities.timewindow import TimeWindow
import numpy as np
from processing import ClusterMetricsCalculatorFactory


class Cluster:
    '''A cluster from one time window containing all metrics used for machine learning.'''

    def __init__(self, time_window_id: Any, cluster_id: Any, cluster_nodes: List[dict], cluster_feature_names: List[str], nr_layer_nodes: int, layer_diversity: int):
        self.time_window_id = time_window_id
        self.cluster_id = cluster_id

        metrics_calculator = ClusterMetricsCalculatorFactory.create_metrics_calculator(cluster_nodes, cluster_feature_names, nr_layer_nodes, layer_diversity)

        self.size = metrics_calculator.get_size()
        self.std_dev = metrics_calculator.get_standard_deviation()
        self.scarcity = metrics_calculator.get_scarcity()
        
        self.importance1 = metrics_calculator.get_importance1()
        self.importance2 = metrics_calculator.get_importance2()

    def get_time_info(self) -> int:
        '''Returns the week of the time tuple str, eg. 25 for "(2014, 25)".'''
        str_tuple = self.time_window_id
        return int(str_tuple.split(',')[1].strip()[:-1])

    def __repr__(self):
        return str(self.__dict__)

    def __str__(self):
        return f"Cluster({self.cluster_id}, " \
        f"{self.size}, {self.std_dev}, {self.scarcity}, " \
        f"{self.importance1}, {self.importance2})"

    @staticmethod
    def create_multiple_from_time_window(time_window: TimeWindow, cluster_feature_names: List[str]) -> Iterable[Cluster]:
        total_layer_nodes = sum([len(nodes) for nodes in time_window.clusters.values()])
        
        layer_diversity = len([nodes for nodes in time_window.clusters.values() if len(nodes) > 0])

        for cluster_nr, cluster_nodes in time_window.clusters.items():
            yield Cluster(time_window.time, cluster_nr, cluster_nodes, cluster_feature_names, total_layer_nodes, layer_diversity)

    @staticmethod
    def create_from_dict(dict_):
        cl = Cluster(0, 0, [], 0, 0, 0)
        cl.__dict__.update(dict_)
        return cl
