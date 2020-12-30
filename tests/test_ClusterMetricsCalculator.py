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


if __name__ == '__main__':
    unittest.main()
