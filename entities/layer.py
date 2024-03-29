from typing import Dict, List, Tuple, Any
import scipy.spatial
from entities.timewindow import TimeWindow
from processing import ClusterMetricsCalculatorFactory


class InternalCluster:
    def __init__(self, cluster_id, cluster_nodes: List[dict], feature_names:List[str], global_cluster_center: Tuple[float], n_layer_nodes: int):
        self.cluster_id = cluster_id

        metrics_calculator = ClusterMetricsCalculatorFactory.create_metrics_calculator(cluster_nodes, feature_names, n_layer_nodes, None)

        self.size = metrics_calculator.get_size()
        self.relative_size = metrics_calculator.get_importance1()
        self.center = metrics_calculator.get_center()

        if self.size > 0:
            self.global_center_distance = scipy.spatial.distance.euclidean(self.center, global_cluster_center)
        else:
            self.global_center_distance = 0

    @staticmethod
    def create_many_from_cluster_nodes(clusters: Dict[str, List[dict]], feature_names: List[str], global_cluster_centers: Dict[str, Tuple[float]]) -> List['InternalCluster']:
        res_clusters = []
        total_layer_nodes = sum([len(nodes) for nodes in clusters.values()])

        for key, value in clusters.items():

            # ignore noise as it contains no meaningful cluster information
            if key == '-1':
                continue

            res_clusters.append(InternalCluster(key, value, feature_names, global_cluster_centers[key], total_layer_nodes))
        return res_clusters


class Layer:
    '''Represents metrics for one layer for a single time window.'''
    def __init__(self, time_window_id: Any, clusters: List[InternalCluster]):
        self.time_window_id = time_window_id

        active_clusters = [c for c in clusters if c.size > 0]
                
        self.n_nodes = sum([c.size for c in clusters])
        self.n_clusters = len(active_clusters)

        self.relative_cluster_sizes = self.get_relative_cluster_sizes(active_clusters)
        self.cluster_size_agg_metrics = self.get_size_min_max_avg_sum(active_clusters)
        self.cluster_relative_size_agg_metrics = self.get_relative_size_min_max_avg_sum(active_clusters)
        self.entropy = self.get_entropy(active_clusters)

        self.centers = [c.center for c in active_clusters]
        self.distances_from_global_centers = self.get_distances_from_global_center(active_clusters)
        self.cluster_center_distance_agg_metrics = self.get_center_distance_min_max_avg_sum(active_clusters)

    def get_size_min_max_avg_sum(self, clusters: List[InternalCluster]) -> dict:
        '''Returns min, max, avg, and sum of the cluster's absolute sizes.'''
        if len(clusters) == 0:
            return {'min':0, 'max':0, 'avg':0, 'sum':0}

        min_ = clusters[0].size
        max_ = clusters[0].size
        sum_ = 0

        for c in clusters:
            value = c.size
            min_ = min(min_, value)
            max_ = max(max_, value)
            sum_ += value
        avg_ = sum_ / len(clusters)
        return {'min': min_, 'max': max_, 'avg': avg_, 'sum': sum_}
        
    def get_relative_size_min_max_avg_sum(self, clusters: List[InternalCluster]) -> dict:
        '''Returns min, max, avg, and sum of the cluster's relative sizes.'''
        if len(clusters) == 0:
            return {'min':0, 'max':0, 'avg':0, 'sum':0}

        min_ = clusters[0].relative_size
        max_ = clusters[0].relative_size
        sum_ = 0

        for c in clusters:
            value = c.relative_size
            min_ = min(min_, value)
            max_ = max(max_, value)
            sum_ += value
        avg_ = sum_ / len(clusters)
        return {'min': min_, 'max': max_, 'avg': avg_, 'sum': sum_}

    def get_center_distance_min_max_avg_sum(self, clusters: List[InternalCluster]) -> dict:
        '''Returns min, max, avg, and sum of the cluster's center distances.'''
        if len(clusters) == 0:
            return {'min':0, 'max':0, 'avg':0, 'sum':0}

        min_ = clusters[0].global_center_distance
        max_ = clusters[0].global_center_distance
        sum_ = 0

        for c in clusters:
            value = c.global_center_distance
            min_ = min(min_, value)
            max_ = max(max_, value)
            sum_ += value
        avg_ = sum_ / len(clusters)
        return {'min': min_, 'max': max_, 'avg': avg_, 'sum': sum_}

    def get_relative_cluster_sizes(self, clusters: List[InternalCluster]):
        return [c.relative_size for c in clusters]
        
    def get_entropy(self, clusters: List[InternalCluster]):
        '''
        Returns the entropy over all clusters C, 
        where P(c_i) is the probability that a node belongs to cluster c_i.
        '''
        return scipy.stats.entropy(self.get_relative_cluster_sizes(clusters), base=2)

    def get_distances_from_global_center(self, clusters: List[InternalCluster]):
        return [cluster.global_center_distance for cluster in clusters]

    def __repr__(self):
        return str(self.__dict__)

    def __str__(self):
        return f"Layer({self.time_window_id}, " \
        f"{self.n_nodes}, {self.n_clusters}, {self.relative_cluster_sizes}, " \
        f"{self.entropy}, {self.centers}, {self.distances_from_global_centers})"

    @staticmethod
    def create_from_time_window(time_window: TimeWindow, feature_names:List[str], global_cluster_centers: Dict[str, Tuple[float]]) -> 'Layer':
        clusters: List[InternalCluster] = InternalCluster.create_many_from_cluster_nodes(time_window.clusters, feature_names, global_cluster_centers)
        return Layer(time_window.time, clusters)
    
    @staticmethod
    def create_from_dict(dict_) -> 'Layer':
        l = Layer(0, [])
        l.__dict__.update(dict_)
        return l