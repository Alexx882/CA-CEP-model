from typing import Dict, List, Tuple, Any
import scipy.spatial
from entities.timewindow import TimeWindow


class InternalCluster:
    def __init__(self, cluster_id, cluster_nodes: List[dict], feature_names:List[str], global_cluster_center: Tuple[float]):
        self.cluster_id = cluster_id
        self.size = len(cluster_nodes)
        self.global_center_distance = scipy.spatial.distance.euclidean(self.get_current_cluster_center(cluster_nodes, feature_names), global_cluster_center)

    def _convert_feature_to_float(self, feature_value) -> float:
        return float(feature_value if feature_value is not "" else 0)

    def get_current_cluster_center(self, nodes, features) -> ('x', 'y'):
        if len(features) == 1:
            values = [self._convert_feature_to_float(node[features[0]]) for node in nodes]
            return (sum(values)/len(values), 0)
        
        if len(features) == 2:
            x = [self._convert_feature_to_float(node[features[0]]) for node in nodes]
            y = [self._convert_feature_to_float(node[features[1]]) for node in nodes]
            centroid = (sum(x) / len(nodes), sum(y) / len(nodes))
            return centroid

    @staticmethod
    def create_many_from_cluster_nodes(clusters: Dict[str, List[dict]], feature_names: List[str], global_cluster_centers: Dict[str, Tuple[float]]) -> List['InternalCluster']:
        res_clusters = []
        for key, value in clusters.items():
            res_clusters.append(InternalCluster(key, value, feature_names, global_cluster_centers[key]))
        return res_clusters


class Layer:
    '''Represents metrics for one layer for a single time window.'''
    def __init__(self, time_window_id: Any, clusters: List[InternalCluster]):
        self.time_window_id = time_window_id
                
        self.relative_cluster_sizes = self.get_relative_cluster_sizes(clusters)
        self.entropy = self.get_entropy(clusters)
        self.distances_from_global_centers = self.get_distances_from_global_center(clusters)

    def get_relative_cluster_sizes(self, clusters: List[InternalCluster]):
        total_size = sum([cluster.size for cluster in clusters])
        return [cluster.size / total_size for cluster in clusters]
        
    def get_entropy(self, clusters: List[InternalCluster]):
        '''
        Returns the entropy over all clusters C, 
        where P(c_i) is the probability that a node belongs to cluster c_i.
        '''
        return scipy.stats.entropy(self.get_relative_cluster_sizes(clusters), base=2)

    def get_distances_from_global_center(self, clusters: List[InternalCluster]):
        return [cluster.global_center_distance for cluster in clusters]

    @staticmethod
    def create_from_time_window(time_window: TimeWindow, feature_names:List[str], global_cluster_centers: Dict[str, Tuple[float]]) -> 'Layer':
        clusters: List[InternalCluster] = InternalCluster.create_many_from_cluster_nodes(clusters, feature_names, global_cluster_centers)
        return Layer(time_window.time, clusters)
    