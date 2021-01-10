import unittest
import sys
for path in ['../', './']:
    sys.path.insert(1, path)

# python -m unittest discover
from entities import Layer, TimeWindow
from entities.layer import InternalCluster


from typing import Any, Tuple, List
from datetime import date, datetime
import json
from math import sqrt
import statistics as stat


class TestInternalCluster(unittest.TestCase):
    def test__init__1d_features__all_values_set(self):
        cluster_nodes = [{"feature":1}, {"feature":1}, {"feature":1}]

        c = InternalCluster("123", cluster_nodes, feature_names=["feature"], global_cluster_center=(1.5,0))
        
        self.assert_internal_cluster(c, '123', 3, .5)

    def test__init__2d_features__all_values_set(self):
        cluster_nodes = [{"feature1":1,'feature2':1}, {"feature1":1,'feature2':1}, {"feature1":1,'feature2':1}]

        c = InternalCluster("123", cluster_nodes, feature_names=["feature1", 'feature2'], global_cluster_center=(1.5,1.5))
        
        # https://www.calculatorsoup.com/calculators/geometry-plane/distance-two-points.php
        self.assert_internal_cluster(c, '123', 3, sqrt(.5))

    def test__get_current_cluster_center__1d(self):
        cluster_nodes = [{"feature":1}, {"feature":2}, {"feature":3}]

        c = InternalCluster("123", cluster_nodes, feature_names=["feature"], global_cluster_center=(2, 0))

        self.assert_internal_cluster(c, '123', 3, 0)
        

    def test__get_current_cluster_center__1d_weighted_result(self):
        cluster_nodes = [{"feature":1}, {"feature":1}, {"feature":3}]

        c = InternalCluster("123", cluster_nodes, feature_names=["feature"], global_cluster_center=(5/3, 0))
        
        self.assert_internal_cluster(c, '123', 3, 0)

    def test__get_current_cluster_center__2d_weighted_result(self):
        cluster_nodes = [{"feature1":1,"feature2":1},
                         {"feature1":1,"feature2":1},
                         {"feature1":2,"feature2":2},
                         {"feature1":3,"feature2":1}]

        c = InternalCluster("123", cluster_nodes, feature_names=["feature1", 'feature2'], global_cluster_center=(1.75, 1.25))
        
        self.assert_internal_cluster(c, '123', 4, 0)

    def assert_internal_cluster(self, actual_cluster: InternalCluster, expected_id, expected_size, expected_distance):
        self.assertEqual(expected_id, actual_cluster.cluster_id)
        self.assertEqual(expected_size, actual_cluster.size)
        self.assertAlmostEqual(expected_distance, actual_cluster.global_center_distance) 


class TestLayer(unittest.TestCase):
    def test__init__single_cluster(self):
        cluster_nodes = list(self._get_timewindow_single_cluster_1d_same_feature().clusters.values())[0]
        c = InternalCluster("123", cluster_nodes, feature_names=["feature"], global_cluster_center=(1,0))

        l = Layer('123', [c])

        self.assert_layer(l, [1], 0, [0])
        
    #region setup methods
    def _get_timewindow_single_cluster_1d_same_feature(self) -> TimeWindow:
        '''Returns a TimeWindow with time=KW1 and three nodes in cluster 1, all feature values = 1.'''
        tw = TimeWindow("KW1", "uc", "uct", "ln")
        tw.add_node_to_cluster("1", {"feature":1})
        tw.add_node_to_cluster("1", {"feature":1})
        tw.add_node_to_cluster("1", {"feature":1})
        return tw

    def _get_timewindow_single_cluster_2d_same_feature(self) -> TimeWindow:
        '''Returns a TimeWindow with time=KW1 and three nodes in cluster 1, all feature1 & feature2 values = 1.'''
        tw = TimeWindow("KW1", "uc", "uct", "ln")
        tw.add_node_to_cluster("1", {"feature1":1, "feature2":1})
        tw.add_node_to_cluster("1", {"feature1":1, "feature2":1})
        tw.add_node_to_cluster("1", {"feature1":1, "feature2":1})
        return tw

    def _get_timewindow_two_clusters_same_feature(self) -> TimeWindow:
        '''
        Returns a TimeWindow with time=KW1 and:
        Three nodes in cluster 1, all feature values = 1.
        Two nodes in cluster 2, all feature values = 2.
        '''
        tw = TimeWindow("KW1", "uc", "uct", "ln")
        tw.add_node_to_cluster("1", {"feature":1})
        tw.add_node_to_cluster("1", {"feature":1})
        tw.add_node_to_cluster("1", {"feature":1})
        tw.add_node_to_cluster("2", {"feature":2})
        tw.add_node_to_cluster("2", {"feature":2})
        return tw

    #endregion setup methods

    def assert_layer(self, actual_layer: Layer, relative_sizes: List[float], entropy: float, center_dist: List[float]):
        self.assertEqual(len(actual_layer.relative_cluster_sizes), len(relative_sizes))
        for i in range(len(relative_sizes)):
            self.assertAlmostEqual(relative_sizes[i], actual_layer.relative_cluster_sizes[i])

        self.assertAlmostEqual(entropy, actual_layer.entropy)

        self.assertEqual(len(actual_layer.distances_from_global_centers), len(center_dist))   
        for i in range(len(center_dist)):
            self.assertAlmostEqual(center_dist[i], actual_layer.distances_from_global_centers[i])


if __name__ == '__main__':
    unittest.main()
