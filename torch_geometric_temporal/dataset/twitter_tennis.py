import io
import json
import numpy as np
from six.moves import urllib
from ..signal import DynamicGraphTemporalSignal

class TwitterTennisDatasetLoader(object):
    """
    A dataset of Twitter mention graphs related to major tennis tournaments from 2017. Nodes are Twitter accounts and edges are mentions between them. Node labels change from day to day as they indicate whether a given node (Twitter account) belongs to a tennis player who played on that day. By setting the 'event_id' parameter, you can choose to load the mention network for Roland-Garros 2017 (rg17) or USOpen 2017 (uo17). Find more details about this data set in the 'Temporal Walk Based Centrality Metric for Graph Streams'. paper.
    """
    def __init__(self, event_id="rg17", N=None):
        self.N = N
        if event_id in ["rg17","uo17"]:
            self.event_id = event_id
        else:
            raise ValueError("Invalid 'event_id'! Choose 'rg17' or 'uo17' to load the Roland-Garros 2017 or the USOpen 2017 Twitter tennis dataset respectively.")
        self._read_web_data()

    def _read_web_data(self):
        fname = "twitter_tennis_%s.json" % self.event_id
        if self.N != None:
            fname = fname.replace(".json","_N%i.json" % self.N)
        url = "https://raw.githubusercontent.com/ferencberes/pytorch_geometric_temporal/developer/dataset/" + fname 
        self._dataset = json.loads(urllib.request.urlopen(url).read())
        #with open("/home/fberes/git/pytorch_geometric_temporal/dataset/"+fname) as f:
        #    self._dataset = json.load(f)

    def _get_edges(self):
        self.edges = []
        for time in range(self._dataset["time_periods"]):
            self.edges.append(np.array(self._dataset[str(time)]["edges"]).T)

    def _get_edge_weights(self):
        self.edge_weights = []
        for time in range(self._dataset["time_periods"]):
            self.edge_weights.append(np.array(self._dataset[str(time)]["weights"]))

    def _get_features(self):
        self.features = []
        for time in range(self._dataset["time_periods"]):
            self.features.append(np.array(self._dataset[str(time)]["X"]))

    def _get_targets(self):
        self.targets = []
        for time in range(self._dataset["time_periods"]):
            self.targets.append(np.array(self._dataset[str(time)]["y"]))

    def get_dataset(self) -> DynamicGraphTemporalSignal:
        """Returning the TennisDataset data iterator.

        Return types:
            * **dataset** *(DynamicGraphTemporalSignal)* - Selected Twitter tennis dataset (Roland-Garros 2017 or USOpen 2017).
        """
        self._get_edges()
        self._get_edge_weights()
        self._get_features()
        self._get_targets()
        dataset = DynamicGraphTemporalSignal(self.edges, self.edge_weights, self.features, self.targets)
        return dataset

