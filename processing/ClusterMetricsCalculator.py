import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
import numpy as np
from scipy.spatial import ConvexHull, qhull, distance
from math import sqrt
from statistics import mean

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
    def get_standard_deviation(self) -> float:
        '''Returns the std dev from the center of the distribution.'''
        pass

    @abstractmethod
    def get_scarcity(self) -> float:
        '''
        Returns the scarcity of the data points regarding the complete range for possible points.
        High scarcity indicates low density.
        '''
        pass

    @abstractmethod
    def get_range(self) -> float:
        '''Returns the range or area of the cluster based on the edge nodes.'''
        pass

    @abstractmethod
    def get_center(self) -> (float, float):
        '''Returns the center of the cluster, output is fixed 2d.'''
        pass

    def get_importance1(self) -> float:
        '''Returns the ratio of cluster_nodes to layer_nodes.'''
        return float(len(self.cluster_nodes)) / self.nr_layer_nodes if len(self.cluster_nodes) > 0 else 0

    def get_importance2(self) -> float:
        '''Returns the inverse of the layer_diversity, where layer_diversity = number of clusters with #nodes > 0.'''
        return 1.0 / self.layer_diversity if len(self.cluster_nodes) > 0 else 0

    def _convert_feature_to_float(self, feature_value) -> float:
        return float(feature_value if feature_value is not "" else 0)

   
class ClusterMetricsCalculator1D(ClusterMetricsCalculator):
    '''Metrics calculator for clusters which were clustered based on 1 feature (1d clustering).'''

    def __init__(self, cluster_nodes: List[dict], cluster_feature_name: str, nr_layer_nodes: int, layer_diversity: int):
        super().__init__(cluster_nodes, nr_layer_nodes, layer_diversity)
        self.feature_values: List[Any] = [self._convert_feature_to_float(node[cluster_feature_name])
                                          for node in cluster_nodes]
        if len(self.feature_values) > 0:
            self.max_value = max(self.feature_values)
            self.min_value = min(self.feature_values)
        else:
            self.max_value = self.min_value = 0

    def get_standard_deviation(self):
        return np.std(self.feature_values) if len(self.feature_values) > 0 else 0

    def get_scarcity(self):
        '''Returns the scarcity as cluster_range / cluster_size, or 0 if len(nodes)=0.'''
        if len(self.feature_values) == 0:
            return 0

        return self.get_range() / self.get_size()

    def get_range(self):
        return float(self.max_value - self.min_value)

    def get_center(self):
        if len(self.feature_values) == 0:
            return (0, 0)
        return (sum(self.feature_values) / len(self.feature_values), 0)


class ClusterMetricsCalculator2D(ClusterMetricsCalculator):
    '''Metrics calculator for clusters which were clustered based on 2 features (2d clustering).'''
    
    def __init__(self, cluster_nodes: List[dict], cluster_feature_names: List[str], nr_layer_nodes: int, layer_diversity: int):
        assert len(cluster_feature_names) == 2, "This class is for 2d cluster results only!"
        super().__init__(cluster_nodes, nr_layer_nodes, layer_diversity)

        self.feature_values: List[Tuple[Any]] = [
             (self._convert_feature_to_float(node[cluster_feature_names[0]]), self._convert_feature_to_float(node[cluster_feature_names[1]]))
             for node in cluster_nodes
             ]

    def get_standard_deviation(self):
        if len(self.feature_values) == 0:
            return 0

        warnings.simplefilter(action='ignore', category=RuntimeWarning)
        std_dist = std_distance(self.feature_values)
        warnings.simplefilter(action='default', category=RuntimeWarning)
        
        if np.isnan(std_dist):
            return 0 # somehow std_dist=nan if all feature values are same with many decimals
            
        return std_dist

    def get_scarcity(self):
        '''Returns the scarcity as cluster_range / cluster_size, or 0 if len(nodes)=0.'''
        if len(self.feature_values) == 0:
            return 0

        if len(self.feature_values) == 1:
            # exactly 1 element gives inf density
            return 0

        range_, twodim = self._get_range()
        if twodim:
            return sqrt(range_ / self.get_size())
        else:
            return range_ / self.get_size()
   
    def _get_range(self):
        twodim = False

        if len(self.feature_values) == 0 or len(self.feature_values) == 1:
            range_ = 0

        elif len(self.feature_values) == 2:
            # cannot calculate area with 2 points - just use 2d distance as range instead
            range_ = float(distance.euclidean(self.feature_values[0], self.feature_values[1]))
        
        else:
            try:
                # calculate range as 2d area
                points = self._get_polygon_border_points(self.feature_values)
                range_ = self._calc_polygon_area(points)
                # twodim must be known when calculating scarcity
                twodim = True

            except qhull.QhullError as err:
                # possible reasons that there is no hull with real area:
                # 1. all points are at the same location
                # 2. all points have the same x or y coordinates (lie on one hori/vert line)
                points = np.asarray(self.feature_values)
                same_x = len(set(points[:,0])) == 1
                if same_x:
                    # use only y feature
                    features = points[:,1]
                    range_ = max(features) - min(features)

                same_y = len(set(points[:,1])) == 1
                if same_y:
                    # use only x feature
                    features = points[:,0]
                    range_ = max(features) - min(features)

                if not same_x and not same_y:
                    # assume linear distribution of nodes
                    points = np.asarray(list(set(self.feature_values)))
                    min_ = min(points[:,0]), min(points[:,1])
                    max_ = max(points[:,0]), max(points[:,1])
                    range_ = float(distance.euclidean(min_, max_))

        return (range_, twodim)
    
    def get_range(self):
        return self._get_range()[0]

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

    def get_center(self):
        if len(self.feature_values) == 0:
            return (0, 0)
            
        x = [f[0] for f in self.feature_values]
        y = [f[1] for f in self.feature_values] 
        centroid = (sum(x) / len(self.feature_values), sum(y) / len(self.feature_values))
        return centroid


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
