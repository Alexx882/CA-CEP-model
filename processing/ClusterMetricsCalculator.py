from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np

class ClusterMetricsCalculator(ABC):
    def __init__(self, cluster_nodes: List[dict], nr_layer_nodes: int, layer_diversity: int):
        self.cluster_nodes = cluster_nodes
        
        self.nr_layer_nodes = nr_layer_nodes
        self.layer_diversity = layer_diversity

    def get_size(self) -> int:
        '''Returns the size of the cluster'''
        return len(self.cluster_nodes)

    @abstractmethod
    def get_variance(self):
        pass

    @abstractmethod
    def get_density(self):
        pass

    def get_importance1(self):
        return float(len(self.cluster_nodes)) / self.nr_layer_nodes if len(self.cluster_nodes) > 0 else 0

    def get_importance2(self):
        return 1.0 / self.layer_diversity if len(self.cluster_nodes) > 0 else 0

   
class ClusterMetricsCalculator1D(ClusterMetricsCalculator):
    def __init__(self, cluster_nodes: List[dict], cluster_feature_name: str, nr_layer_nodes: int, layer_diversity: int):
        super().__init__(cluster_nodes, nr_layer_nodes, layer_diversity)
        self.feature_values = [node[cluster_feature_name] for node in cluster_nodes]

    def get_variance(self):
        return np.var(self.feature_values) if len(self.feature_values) > 0 else 0

    def get_density(self):
        '''Returns the density as cluster_range / # cluster_nodes, or 0 if len(nodes)=0.'''
        if len(self.feature_values) == 0:
            return 0

        range_ = max(self.feature_values) - min(self.feature_values)
        return float(range_) / len(self.feature_values)


class ClusterMetricsCalculator2D(ClusterMetricsCalculator):
    pass


class ClusterMetricsCalculatorFactory:
    @staticmethod
    def create_metrics_calculator(cluster_nodes: List[dict], cluster_feature_names: List[str], nr_layer_nodes: int, layer_diversity: int) -> ClusterMetricsCalculator:
        """
        This factory creates a class which contains metrics about a single cluster based on 
        its nodes, feature values, its layer total node number and its layer diversity.

        :param cluster_nodes: all nodes from the cluster
        :param cluster_feature_names: all field names which where used during clustering
        :param nr_layer_nodes: the number of total layer nodes
        :param layer_diversity: the diversity of the layer calculated as: number of clusters with nodes > 0
        """
        if isinstance(cluster_feature_names, str):
            return ClusterMetricsCalculator1D(cluster_nodes, cluster_feature_names, nr_layer_nodes, layer_diversity)
        if len(cluster_feature_names) == 1:
            return ClusterMetricsCalculator1D(cluster_nodes, cluster_feature_names[0], nr_layer_nodes, layer_diversity)
            
        if len(cluster_feature_names) == 2:
            return ClusterMetricsCalculator2D(cluster_nodes, cluster_feature_names[0], nr_layer_nodes, layer_diversity)

