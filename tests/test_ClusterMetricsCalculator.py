import unittest
import sys
for path in ['../', './']:
    sys.path.insert(1, path)

# python -m unittest discover
from processing import ClusterMetricsCalculator2D


class TestClusterMetricsCalculator(unittest.TestCase):
    
    def test__get_standard_deviation__same_points_many_decimals__zero_and_not_nan(self):
        nodes = [{'f1': -8.58564, 'f2': 41.148567},
                {'f1': -8.58564, 'f2': 41.148567},
                {'f1': -8.58564, 'f2': 41.148567},
                {'f1': -8.58564, 'f2': 41.148567},
                {'f1': -8.58564, 'f2': 41.148567},
                {'f1': -8.58564, 'f2': 41.148567},
                {'f1': -8.58564, 'f2': 41.148567},
                {'f1': -8.58564, 'f2': 41.148567},
                {'f1': -8.58564, 'f2': 41.148567}]
        calc = ClusterMetricsCalculator2D(nodes, ['f1','f2'], len(nodes), 1)
        
        self.assertAlmostEqual(0, calc.get_standard_deviation())

    def test__get_range__almost_linear_distribution_in_2d__euclidean_distance(self):
        l = [(-8.657802, 41.160978), (-8.65782, 41.160969), (-8.657838, 41.16096)]
        nodes = [{'f1': e[0], 'f2': e[1]} for e in l]
        
        calc = ClusterMetricsCalculator2D(nodes, ['f1','f2'], len(nodes), 1)
        
        # https://www.calculatorsoup.com/calculators/geometry-plane/distance-two-points.php
        self.assertAlmostEqual(4.0E-5, calc.get_range(), 5)
    

if __name__ == '__main__':
    unittest.main()
