import json
from typing import List, Dict, NewType, Any
from datetime import date, datetime

class TimeWindow:
    '''
    A time slice for a single layer containing all nodes for that time.

    :param time: The tag indicating the time
    :param layer_name: The name of the layer the nodes belong to
    '''

    def __init__(self, time: Any = None, use_case: str = None, use_case_table: str = None, layer_name: str = None,
                 time_slice_dict: Dict = None, from_db = False):
        self.time = str(time)
        self.use_case = use_case
        self.use_case_table = use_case_table
        self.layer_name = layer_name
        self.clusters: Dict[str, List[dict]] = {}

        if time_slice_dict is not None:
            self.from_serializable_dict(time_slice_dict, from_db)

    def add_node_to_cluster(self, cluster_label: str, node):
        # only string keys can be stored in json
        cluster_label = str(cluster_label)

        if cluster_label not in self.clusters:
            self.clusters[cluster_label] = []

        # node = self._get_unique_id(node)
        self.clusters[cluster_label].append(node)

    def get_nodes_for_cluster(self, cluster_label: str):
        if cluster_label in self.clusters:
            return self.clusters[cluster_label]
        else:
            return []
        
    def _get_unique_id(self, node : Dict) -> Dict:
        '''Returns a new dict with the unique id only.'''
        uid_key = 'UniqueID'
        if uid_key in node:
            return {uid_key: node[uid_key]}


    def to_serializable_dict(self, for_db=False) -> Dict:
        return {
            "time": self.time,
            "use_case": self.use_case,
            "use_case_table": self.use_case_table,
            'layer_name': self.layer_name,
            "clusters": json.dumps(self.clusters) if for_db else self.clusters
        }

    def from_serializable_dict(self, dict: Dict, from_db=False):
        self.time = dict["time"]
        self.use_case = dict["use_case"]
        self.use_case_table = dict["use_case_table"]
        self.layer_name = dict['layer_name']
        self.clusters = json.loads(dict['clusters']) if from_db else dict['clusters']

    @staticmethod
    def create_from_serializable_dict(dict: Dict, from_db=False):
        ts = TimeWindow()
        ts.from_serializable_dict(dict, from_db)
        return ts

    def __repr__(self):
        return json.dumps(self.to_serializable_dict())

    def __str__(self):
        return f"TimeWindow({self.__repr__()})"
