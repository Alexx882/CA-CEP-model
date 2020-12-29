from __future__ import annotations
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
import numpy as np
from scipy.spatial import ConvexHull, qhull, distance
from math import sqrt

warnings.simplefilter(action='ignore', category=UserWarning)
# UserWarning: geopandas not available. Some functionality will be disabled.
from pointpats.centrography import std_distance 
warnings.simplefilter(action='default', category=UserWarning)


class ClusterMetricsCalculator(ABC):
    def __init__(self, cluster_nodes: List[dict], nr_layer_nodes: int, layer_diversity: int):
        self.cluster_nodes = cluster_nodes
        
        self.nr_layer_nodes = nr_layer_nodes
        self.layer_diversity = layer_diversity

    def get_size(self) -> int:
        '''Returns the size of the cluster.'''
        return len(self.cluster_nodes)

    @abstractmethod
    def get_variance(self) -> float:
        '''Returns the difference from the center for the distribution.'''
        pass

    @abstractmethod
    def get_scarcity(self) -> float:
        '''
        Returns the scarcity of the data points regarding the complete range for possible points.
        High scarcity indicates low density.
        '''
        pass

    def get_importance1(self) -> float:
        '''Returns the ratio of cluster_nodes to layer_nodes.'''
        return float(len(self.cluster_nodes)) / self.nr_layer_nodes if len(self.cluster_nodes) > 0 else 0

    def get_importance2(self) -> float:
        '''Returns the inverse of the layer_diversity, where layer_diversity = number of clusters with #nodes > 0.'''
        return 1.0 / self.layer_diversity if len(self.cluster_nodes) > 0 else 0

   
class ClusterMetricsCalculator1D(ClusterMetricsCalculator):
    '''Metrics calculator for clusters which were clustered based on 1 feature (1d clustering).'''

    def __init__(self, cluster_nodes: List[dict], cluster_feature_name: str, nr_layer_nodes: int, layer_diversity: int):
        super().__init__(cluster_nodes, nr_layer_nodes, layer_diversity)
        self.feature_values: List[Any] = [node[cluster_feature_name] for node in cluster_nodes]

    def get_variance(self):
        # TODO check if std is better
        return np.var(self.feature_values) if len(self.feature_values) > 0 else 0

    def get_scarcity(self):
        '''Returns the scarcity as cluster_range / cluster_size, or 0 if len(nodes)=0.'''
        if len(self.feature_values) == 0:
            return 0

        range_ = max(self.feature_values) - min(self.feature_values)
        return float(range_) / self.get_size()


class ClusterMetricsCalculator2D(ClusterMetricsCalculator):
    '''Metrics calculator for clusters which were clustered based on 2 features (2d clustering).'''

    def __init__(self, cluster_nodes: List[dict], cluster_feature_names: List[str], nr_layer_nodes: int, layer_diversity: int):
        assert len(cluster_feature_names) == 2, "This class is for 2d cluster results only!"
        super().__init__(cluster_nodes, nr_layer_nodes, layer_diversity)

        self.feature_values: List[Tuple[Any]] = [
             (node[cluster_feature_names[0]], node[cluster_feature_names[1]])
             for node in cluster_nodes
             ]

    def get_variance(self):
        if len(self.feature_values) == 0:
            return 0

        return std_distance(self.feature_values)

    def get_scarcity(self):
        '''Returns the scarcity as cluster_range / cluster_size, or 0 if len(nodes)=0.'''
        if len(self.feature_values) == 0 or len(self.feature_values) == 1:
            return 0

        if len(self.feature_values) == 2:
            # cannot calculate area with 2 points
            range_ = distance.euclidean(self.feature_values[0], self.feature_values[1])
            return float(range_) / self.get_size()

        # calculate range as 2d area
        try:
            points = self._get_polygon_border_points(self.feature_values)
            range_ = self._calc_polygon_area(points)
        except qhull.QhullError as err:
            # possibly because all points are at the same location
            # therefore calculating a hull with real area is not possible
            print(f"Error while calculating the 2d area for density")
            # this results in infinite density -> 0 scarcity
            return 0

        # use sqrt to compare with 1d density
        return sqrt(float(range_) / self.get_size())

    def _get_polygon_border_points(self, points: List[List[float]]) -> 'np.array':
        points = np.asarray(points)
        hull = ConvexHull(points)
        return points[hull.vertices]

    def _calc_polygon_area(self, border_points: 'np.array') -> float:
        x: 'np.array' = border_points[:,0]
        y: 'np.array' = border_points[:,1]
        # https://en.wikipedia.org/wiki/Shoelace_formula
        area = 0.5 * np.abs(np.dot(x, np.roll(y,1)) - np.dot(y, np.roll(x,1)))
        return float(area)
    

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
            return ClusterMetricsCalculator2D(cluster_nodes, cluster_feature_names, nr_layer_nodes, layer_diversity)
