import unittest
import sys
for path in ['../', './']:
    sys.path.insert(1, path)

# python -m unittest discover
from entities import Cluster, TimeWindow

from typing import Any, Tuple
from datetime import date, datetime
import json
from math import sqrt
import statistics as stat


class TestCluster(unittest.TestCase):
    def test__init__single_cluster__all_values_set(self):
        tw, centers = self._get_timewindow_single_cluster_same_feature()

        c = Cluster("time_abc", "clusterId 1", list(tw.clusters.values())[0], "feature", nr_layer_nodes=3, layer_diversity=1, global_cluster_center=centers['1'])
        
        self.assertEqual("time_abc", c.time_window_id)
        self.assertEqual("clusterId 1", c.cluster_id)
        self.assert_cluster((3, 0, 0, 1, 1), c)
        self.assertEqual(c.range_, 0)
        self.assertEqual(c.center, (1,0))
        self.assertEqual(c.global_center_distance, 0)

    def test__create_multiple_from_time_window__single_cluster__all_values_set(self):
        tw, centers = self._get_timewindow_single_cluster_same_feature()
        
        clusters = list(Cluster.create_multiple_from_time_window(tw, "feature", centers))
        self.assertEqual(1, len(clusters))
        c = clusters[0]

        self.assertEqual("KW1", c.time_window_id)
        self.assertEqual("1", c.cluster_id)
        self.assert_cluster((3, 0, 0, 1, 1), c)

    def test__create_multiple_from_time_window__two_clusters__correct_time_id_cluster_id(self):
        tw, centers = self._get_timewindow_two_clusters_same_feature()
        
        clusters = Cluster.create_multiple_from_time_window(tw, "feature", centers)
        expected = [("KW1", "1"), ("KW1", "2")]

        for c, exp in zip(clusters, expected):
            self.assertEqual(exp[0], c.time_window_id)
            self.assertEqual(exp[1], c.cluster_id)

    def test__create_multiple_from_time_window__two_clusters_same_features__correct_calculation(self):
        tw, centers = self._get_timewindow_two_clusters_same_feature()

        clusters = Cluster.create_multiple_from_time_window(tw, "feature", centers)
        expected = [(3, 0, 0, 3/5, 1/2), (2, 0, 0, 2/5, 1/2)]

        for c, exp in zip(clusters, expected):
            self.assert_cluster(exp, c)

    def test__create_multiple_from_time_window__two_clusters_same_features_and_feature_names_list__correct_calculation(self):
        tw, centers = self._get_timewindow_two_clusters_same_feature()

        clusters = Cluster.create_multiple_from_time_window(tw, ["feature"], centers)
        expected = [(3, 0, 0, 3/5, 1/2), (2, 0, 0, 2/5, 1/2)]

        for c, exp in zip(clusters, expected):
            self.assert_cluster(exp, c)

    def test__create_multiple_from_time_window__two_clusters_different_features__correct_calculation(self):
        tw = TimeWindow("CW1", "uc", "uct", "ln")
        tw.add_node_to_cluster("1", {"feature":1})
        tw.add_node_to_cluster("1", {"feature":2})
        tw.add_node_to_cluster("1", {"feature":3})
        tw.add_node_to_cluster("2", {"feature":70})
        tw.add_node_to_cluster("2", {"feature":75})

        centers = {'1':(2,0),'2':(72.5,0)}

        clusters = Cluster.create_multiple_from_time_window(tw, "feature", centers)
        # variance for stddev calculated with: http://www.alcula.com/calculators/statistics/variance/
        expected = [(3, sqrt(2.0/3), 2.0/3, 3/5, 1/2), (2, sqrt(6.25), 5.0/2, 2/5, 1/2)]

        for cluster, exp in zip(clusters, expected):
            self.assert_cluster(exp, cluster)

    def test__create_multiple_from_time_window__empty_cluster__all_zero_for_empty_cluster(self):
        tw = TimeWindow("CW1", "uc", "uct", "ln")
        tw.add_node_to_cluster("1", {"feature":1})
        tw.add_node_to_cluster("1", {"feature":2})
        tw.add_node_to_cluster("1", {"feature":3})
        tw.add_node_to_cluster("2", {"feature":70})
        tw.add_node_to_cluster("2", {"feature":75})
        tw.clusters["3"] = []

        centers = {'1':(2,0),'2':(72.5,0),'3':(0,0)}

        clusters = Cluster.create_multiple_from_time_window(tw, "feature", centers)
        expected = [(3, sqrt(2.0/3), 2.0/3, 3/5, 1/2), # diversity is still 2 as len=0 is ignored
                    (2, sqrt(6.25), 5.0/2, 2/5, 1/2),
                    (0, 0, 0, 0, 0)] # len 0 -> everything 0

        for cluster, exp in zip(clusters, expected):
            self.assert_cluster(exp, cluster)

    def test__create_multiple_from_time_window__2d_clustering_single_feature_value__no_stddev_no_scarcity(self):
        tw = TimeWindow("CW1", "uc", "uct", "ln")
        tw.add_node_to_cluster("1", {"f1":1, "f2":1})
        tw.add_node_to_cluster("1", {"f1":1, "f2":1})
        tw.add_node_to_cluster("1", {"f1":1, "f2":1})
        tw.add_node_to_cluster("2", {"f1":70, "f2":70})
        tw.add_node_to_cluster("2", {"f1":70, "f2":70})
        
        centers = {'1':(1,1),'2':(70,70)}

        clusters = Cluster.create_multiple_from_time_window(tw, ["f1", "f2"], centers)
        expected = [(3, 0, 0, 3/5, 1/2), (2, 0, 0, 2/5, 1/2)]

        for cluster, exp in zip(clusters, expected):
            self.assert_cluster(exp, cluster)

    def test__create_multiple_from_time_window__2d_clustering__correct_stddev_and_scarcity(self):
        tw = TimeWindow("CW1", "uc", "uct", "ln")
        tw.add_node_to_cluster("1", {"f1":1, "f2":1})
        tw.add_node_to_cluster("1", {"f1":2, "f2":1})
        tw.add_node_to_cluster("1", {"f1":1, "f2":3})
        tw.add_node_to_cluster("2", {"f1":70, "f2":70})
        tw.add_node_to_cluster("2", {"f1":72, "f2":75})

        centers = {'1':(4/3,5/3),'2':(71,72.5)}

        clusters = Cluster.create_multiple_from_time_window(tw, ["f1", "f2"], centers)
        # stddev calculated manually as in: https://glenbambrick.com/tag/standard-distance/
        # area of the polygon calculated with: https://www.mathopenref.com/coordpolygonareacalc.html
        expected = [(3, sqrt(2/9+8/9), sqrt(1/3), 3/5, 1/2), (2, sqrt(7.25), sqrt(2*2+5*5)/2, 2/5, 1/2)] 

        for cluster, exp in zip(clusters, expected):
            self.assert_cluster(exp, cluster)
                        
    def test__create_multiple_from_time_window__2d_clustering_complex__correct_stddev_and_scarcity(self):
        tw = TimeWindow("CW1", "uc", "uct", "ln")
        tw.add_node_to_cluster("1", {"f1":0, "f2":0})
        tw.add_node_to_cluster("1", {"f1":1, "f2":3})
        tw.add_node_to_cluster("1", {"f1":3, "f2":2})
        tw.add_node_to_cluster("1", {"f1":0, "f2":2})
        tw.add_node_to_cluster("1", {"f1":1, "f2":2}) # inside the convex hull
        tw.add_node_to_cluster("1", {"f1":2, "f2":2}) # inside the convex hull
        tw.add_node_to_cluster("1", {"f1":2, "f2":1})

        centers = {'1':((1+3+1+2+2)/7,(3+2+2+2+2+1)/7)}

        clusters = Cluster.create_multiple_from_time_window(tw, ["f1", "f2"], centers)

        # stddev calculated manually as in: https://glenbambrick.com/tag/standard-distance/
        X = [0,1,3,0,1,2,2]
        Y = [0,3,2,2,2,2,1]
        x_mean = stat.mean(X)
        y_mean = stat.mean(Y)
        sum_x = 0
        for x in X:
            sum_x += (x - x_mean)**2
        sum_y = 0
        for y in Y:
            sum_y += (y - y_mean)**2
        sd = sqrt(sum_x/7 + sum_y/7)

        # area of the polygon calculated with: https://www.mathopenref.com/coordpolygonareacalc.html
        area = 5
        scarcity = sqrt(area / 7)

        expected = [[7, sd, scarcity, 1, 1]]

        for cluster, exp in zip(clusters, expected):
            self.assert_cluster(exp, cluster)

    def test__create_multiple_from_time_window__2d_clustering_1d_single_feature_value__correct_calculation(self):
        tw = TimeWindow("CW1", "uc", "uct", "ln")
        tw.add_node_to_cluster("1", {"f1":1, "f2":1})
        tw.add_node_to_cluster("1", {"f1":1, "f2":2})
        tw.add_node_to_cluster("1", {"f1":1, "f2":3})
        tw.add_node_to_cluster("2", {"f1":70, "f2":70})
        tw.add_node_to_cluster("2", {"f1":75, "f2":70})
        tw.add_node_to_cluster("2", {"f1":72, "f2":70})
        tw.add_node_to_cluster("2", {"f1":71, "f2":70})

        centers = {'1':(1,2),'2':((70+75+72+71)/4,70)}

        clusters = Cluster.create_multiple_from_time_window(tw, ["f1", "f2"], centers)
        # variance/stddev calculated as for 1d cluster (as f1/f2 is always the same)
        # scarcity calculated as for 1d cluster 
        expected = [(3, sqrt(2/3), 2/3, 3/7, 1/2), 
                    (4, sqrt(3.5), 5/4, 4/7, 1/2)] 

        for cluster, exp in zip(clusters, expected):
            self.assert_cluster(exp, cluster)

    def test__create_multiple_from_time_window__1d_clusters__correct_cluster_range_and_center(self):
        tw = TimeWindow("CW1", "uc", "uct", "ln")
        tw.add_node_to_cluster("1", {"f1":1})
        tw.add_node_to_cluster("1", {"f1":2})
        tw.add_node_to_cluster("1", {"f1":1})
        tw.add_node_to_cluster("2", {"f1":70})
        tw.add_node_to_cluster("2", {"f1":72})

        centers = {'1':(4/3, 0),'2':(71,0)}

        clusters = Cluster.create_multiple_from_time_window(tw, ["f1"], centers)
        expected = [(1, (4/3,0)), (2, (71,0))] # (range, center)

        for c, exp in zip(clusters, expected):
            self.assertEqual(c.range_, exp[0])
            self.assertEqual(c.center, exp[1])

    def test__create_multiple_from_time_window__2d_clusters__correct_cluster_range_and_center(self):
        tw = TimeWindow("CW1", "uc", "uct", "ln")
        tw.add_node_to_cluster("1", {"f1":1, "f2":1})
        tw.add_node_to_cluster("1", {"f1":2, "f2":1})
        tw.add_node_to_cluster("1", {"f1":1, "f2":3})
        tw.add_node_to_cluster("2", {"f1":70, "f2":70})
        tw.add_node_to_cluster("2", {"f1":72, "f2":75})
        
        centers = {'1':(4/3,5/3),'2':(71,72.5)}

        clusters = Cluster.create_multiple_from_time_window(tw, ["f1","f2"], centers)
        # https://www.triangle-calculator.com/de/?what=vc&a=1&a1=1&3dd=3D&a2=&b=2&b1=1&b2=&c=1&c1=3&c2=&submit=Berechnen&3d=0
        # https://www.calculatorsoup.com/calculators/geometry-plane/distance-two-points.php
        expected = [(1, (4/3,5/3)), (5.385165, (71,72.5))] # (range, center)

        for c, exp in zip(clusters, expected):
            self.assertAlmostEqual(c.range_, exp[0], places=6)
            self.assertEqual(c.center, exp[1])

    def test__create_multiple_from_time_window__2d_clusters__correct_linear_global_center_distance(self):
        tw = TimeWindow("CW1", "uc", "uct", "ln")
        tw.add_node_to_cluster("1", {"f1":1, "f2":2})
        tw.add_node_to_cluster("1", {"f1":1, "f2":2})
        tw.add_node_to_cluster("1", {"f1":1, "f2":2})
        tw.add_node_to_cluster("2", {"f1":70, "f2":70})
        tw.add_node_to_cluster("2", {"f1":72, "f2":75})
        
        centers = {'1':(1,1),'2':(71,70)}

        clusters = Cluster.create_multiple_from_time_window(tw, ["f1","f2"], centers)
        expected = [1, 2.5] # euclidean dist

        for c, exp in zip(clusters, expected):
            self.assertAlmostEqual(c.global_center_distance, exp, places=6)

    def test__create_multiple_from_time_window__1d_clusters__correct_euclidean_global_center_distance(self):
        tw = TimeWindow("CW1", "uc", "uct", "ln")
        tw.add_node_to_cluster("1", {"f1":1})
        tw.add_node_to_cluster("1", {"f1":1})
        tw.add_node_to_cluster("1", {"f1":1})
        tw.add_node_to_cluster("2", {"f1":70})
        tw.add_node_to_cluster("2", {"f1":72})
        
        centers = {'1':(1,0),'2':(70,0)}

        clusters = Cluster.create_multiple_from_time_window(tw, ["f1"], centers)
        # https://www.calculatorsoup.com/calculators/geometry-plane/distance-two-points.php
        expected = [0, 1] # euclidean dist

        for c, exp in zip(clusters, expected):
            self.assertAlmostEqual(c.global_center_distance, exp, places=6)
            
    def test__create_multiple_from_time_window__2d_clusters__correct_euclidean_global_center_distance(self):
        tw = TimeWindow("CW1", "uc", "uct", "ln")
        tw.add_node_to_cluster("1", {"f1":1, "f2":2})
        tw.add_node_to_cluster("1", {"f1":1, "f2":2})
        tw.add_node_to_cluster("1", {"f1":1, "f2":2})
        tw.add_node_to_cluster("2", {"f1":70, "f2":70})
        tw.add_node_to_cluster("2", {"f1":72, "f2":75})
        
        centers = {'1':(0,0),'2':(70,70)}

        clusters = Cluster.create_multiple_from_time_window(tw, ["f1","f2"], centers)
        # https://www.calculatorsoup.com/calculators/geometry-plane/distance-two-points.php
        expected = [2.236068,  2.692582] # euclidean dist

        for c, exp in zip(clusters, expected):
            self.assertAlmostEqual(c.global_center_distance, exp, places=6)


#region setup methods
    def _get_timewindow_single_cluster_same_feature(self) -> TimeWindow:
        '''Returns a TimeWindow with time=KW1 and three nodes in cluster 1, all feature values = 1.'''
        tw = TimeWindow("KW1", "uc", "uct", "ln")
        tw.add_node_to_cluster("1", {"feature":1})
        tw.add_node_to_cluster("1", {"feature":1})
        tw.add_node_to_cluster("1", {"feature":1})

        global_centers = {'1':(1,0)}

        return tw, global_centers

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

        global_centers = {'1':(1,0),'2':(2,0)}

        return tw, global_centers

#endregion setup methods

#region custom asserts
    def assert_cluster(self, expected_values: Tuple[Any], cluster: Cluster):
        """
        Checks if the cluster values equal the expected_values.

        :param expected_values: A tuple (exp_size, exp_stddev, exp_scarcity, exp_import1, exp_import2)
        """
        self.assertEqual(expected_values[0], cluster.size)
        self.assertAlmostEqual(expected_values[1], cluster.std_dev)
        self.assertAlmostEqual(expected_values[2], cluster.scarcity)
        
        self.assertAlmostEqual(expected_values[3], cluster.importance1)
        self.assertAlmostEqual(expected_values[4], cluster.importance2)

#endregion custom asserts


if __name__ == '__main__':
    unittest.main()
