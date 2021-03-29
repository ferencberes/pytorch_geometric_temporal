import numpy as np
import networkx as nx

from torch_geometric_temporal.signal import temporal_signal_split
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from torch_geometric_temporal.signal import DynamicGraphTemporalSignal

from torch_geometric_temporal.dataset import METRLADatasetLoader, PemsBayDatasetLoader
from torch_geometric_temporal.dataset import ChickenpoxDatasetLoader, PedalMeDatasetLoader, WikiMathsDatasetLoader
from torch_geometric_temporal.dataset import TwitterTennisDatasetLoader


def get_edge_array(n_count):
    return np.array([edge for edge in nx.gnp_random_graph(n_count, 0.1).edges()]).T

def generate_signal(snapshot_count, n_count, feature_count):
    edge_indices = [get_edge_array(n_count) for _ in range(snapshot_count)]
    edge_weights = [np.ones(edge_indices[t].shape[1]) for t in range(snapshot_count)]
    features = [np.random.uniform(0,1,(n_count, feature_count)) for _ in range(snapshot_count)]
    return edge_indices, edge_weights, features

def test_dynamic_graph_discrete_signal_real():

    snapshot_count = 250
    n_count = 100
    feature_count = 32

    edge_indices, edge_weights, features = generate_signal(250, 100, 32)

    targets = [np.random.uniform(0,10,(n_count,)) for _ in range(snapshot_count)]

    dataset = DynamicGraphTemporalSignal(edge_indices, edge_weights, features, targets)

    for epoch in range(2):
        for snapshot in dataset:
            assert snapshot.edge_index.shape[0] == 2
            assert snapshot.edge_index.shape[1] == snapshot.edge_attr.shape[0]
            assert snapshot.x.shape == (100, 32)
            assert snapshot.y.shape == (100, )


    
    targets = [np.floor(np.random.uniform(0,10,(n_count,))).astype(int) for _ in range(snapshot_count)]

    dataset = DynamicGraphTemporalSignal(edge_indices, edge_weights, features, targets)

    for epoch in range(2):
        for snapshot in dataset:
            assert snapshot.edge_index.shape[0] == 2
            assert snapshot.edge_index.shape[1] == snapshot.edge_attr.shape[0]
            assert snapshot.x.shape == (100, 32)
            assert snapshot.y.shape == (100, )


def test_static_graph_discrete_signal():
    dataset = StaticGraphTemporalSignal(None, None, [None, None],[None, None])
    for snapshot in dataset:
        assert snapshot.edge_index is None
        assert snapshot.edge_attr is None
        assert snapshot.x is None
        assert snapshot.y is None

def test_dynamic_graph_discrete_signal():
    dataset = DynamicGraphTemporalSignal([None, None], [None, None], [None, None],[None, None])
    for snapshot in dataset:
        assert snapshot.edge_index is None
        assert snapshot.edge_attr is None
        assert snapshot.x is None
        assert snapshot.y is None

def test_static_graph_discrete_signal_typing():
    dataset = StaticGraphTemporalSignal(None, None, [np.array([1])],[np.array([2])])
    for snapshot in dataset:
        assert snapshot.edge_index is None
        assert snapshot.edge_attr is None
        assert snapshot.x.shape == (1,)
        assert snapshot.y.shape == (1,)

def test_chickenpox():
    loader = ChickenpoxDatasetLoader()
    dataset = loader.get_dataset()
    for epoch in range(3):
        for snapshot in dataset:
            assert snapshot.edge_index.shape == (2, 102)
            assert snapshot.edge_attr.shape == (102, )
            assert snapshot.x.shape == (20, 4)
            assert snapshot.y.shape == (20, )
            
def test_pedalme():
    loader = PedalMeDatasetLoader()
    dataset = loader.get_dataset()
    for epoch in range(3):
        for snapshot in dataset:
            assert snapshot.edge_index.shape == (2, 225)
            assert snapshot.edge_attr.shape == (225, )
            assert snapshot.x.shape == (15, 4)
            assert snapshot.y.shape == (15, )
            
def test_wiki():
    loader = WikiMathsDatasetLoader()
    dataset = loader.get_dataset()
    for epoch in range(3):
        for snapshot in dataset:
            snapshot.edge_index.shape == (2, 27079)
            snapshot.edge_attr.shape == (27079, )
            snapshot.x.shape == (1068, 8)
            snapshot.y.shape == (1068, )

def test_metrla():
    loader = METRLADatasetLoader(raw_data_dir="/tmp/")
    dataset = loader.get_dataset()
    for epoch in range(3):
        for snapshot in dataset:
            assert snapshot.edge_index.shape == (2, 1722)
            assert snapshot.edge_attr.shape == (1722, )
            assert snapshot.x.shape == (207, 2, 12)
            assert snapshot.y.shape == (207, 12)

def test_metrla_task_generator():
    loader = METRLADatasetLoader(raw_data_dir="/tmp/")
    dataset = loader.get_dataset(num_timesteps_in=6, num_timesteps_out=5)
    for epoch in range(3):
        for snapshot in dataset:
            assert snapshot.edge_index.shape == (2, 1722)
            assert snapshot.edge_attr.shape == (1722, )
            assert snapshot.x.shape == (207, 2, 6)
            assert snapshot.y.shape == (207, 5)

def test_pemsbay():
    loader = PemsBayDatasetLoader(raw_data_dir="/tmp/")
    dataset = loader.get_dataset()
    for epoch in range(3):
        for snapshot in dataset:
            assert snapshot.edge_index.shape == (2, 2694)
            assert snapshot.edge_attr.shape == (2694, )
            assert snapshot.x.shape == (325, 2, 12)
            assert snapshot.y.shape == (325, 2, 12)

def test_pemsbay_task_generator():
    loader = PemsBayDatasetLoader(raw_data_dir="/tmp/")
    dataset = loader.get_dataset(num_timesteps_in=6, num_timesteps_out=5)
    for epoch in range(3):
        for snapshot in dataset:
            assert snapshot.edge_index.shape == (2, 2694)
            assert snapshot.edge_attr.shape == (2694, )
            assert snapshot.x.shape == (325, 2, 6)
            assert snapshot.y.shape == (325, 2, 5)

def check_tennis_data(event_id, total_nodes, edges_in_snapshots):
    for N, edge_cnt in edges_in_snapshots.items():
        loader = TwitterTennisDatasetLoader(event_id, N)
        dataset = loader.get_dataset()
        node_cnt = total_nodes if N == None else N
        for epoch in range(3):
            i = 0
            for snapshot in dataset:
                assert snapshot.edge_index.shape == (2, edge_cnt[i])
                assert snapshot.edge_attr.shape == (edge_cnt[i], )
                assert snapshot.x.shape == (node_cnt,2)
                assert snapshot.y.shape == (node_cnt,)
                i += 1
            
def test_twitter_tennis_rg17():
    edges_in_snapshots = {
        None : [11514, 13483, 13392, 14987, 12316, 12665, 14216, 13153, 11522, 12064, 16451, 12049, 18346, 13327, 40117],
        1000 : [1740, 2285, 2177, 2394, 2277, 2312, 2073, 2287, 1990, 1633, 2405, 1670, 1903, 1354, 1695],
        5000 : [4493, 5561, 5345, 5731, 5130, 5292, 5255, 5438, 4642, 4296, 6103, 4278, 5474, 3921, 6071]
    }
    check_tennis_data("rg17", 74983, edges_in_snapshots)
    
def test_twitter_tennis_uo17():
    edges_in_snapshots = {
        None : [23670, 24229, 20012, 19017, 16411, 18020, 19506, 24688, 16553, 28957, 23999, 25276, 27282, 28094],
        1000 : [2580, 2327, 2474, 2544, 2408, 2533, 2677, 2537, 1823, 2560, 2115, 1929, 1869, 1659],
        5000 : [7373, 6623, 6606, 6373, 5768, 6399, 6672, 6941, 4932, 7501, 6355, 5868, 5739, 5380]
    }
    check_tennis_data("uo17", 99190, edges_in_snapshots)

def test_discrete_train_test_split_static():
    loader = ChickenpoxDatasetLoader()
    dataset = loader.get_dataset()
    train_dataset, test_dataset = temporal_signal_split(dataset, 0.8)

    for epoch in range(2):
        for snapshot in train_dataset:
            assert snapshot.edge_index.shape == (2, 102)
            assert snapshot.edge_attr.shape == (102, )
            assert snapshot.x.shape == (20, 4)
            assert snapshot.y.shape == (20, )

    for epoch in range(2):
        for snapshot in test_dataset:
            assert snapshot.edge_index.shape == (2, 102)
            assert snapshot.edge_attr.shape == (102, )
            assert snapshot.x.shape == (20, 4)
            assert snapshot.y.shape == (20, )


def test_discrete_train_test_split_dynamic():

    snapshot_count = 250
    n_count = 100
    feature_count = 32

    edge_indices, edge_weights, features = generate_signal(250, 100, 32)

    targets = [np.random.uniform(0,10,(n_count,)) for _ in range(snapshot_count)]

    dataset = DynamicGraphTemporalSignal(edge_indices, edge_weights, features, targets)


    train_dataset, test_dataset = temporal_signal_split(dataset, 0.8)


    for epoch in range(2):
        for snapshot in test_dataset:
            assert snapshot.edge_index.shape[0] == 2
            assert snapshot.edge_index.shape[1] == snapshot.edge_attr.shape[0]
            assert snapshot.x.shape == (100, 32)
            assert snapshot.y.shape == (100, )

    for epoch in range(2):
        for snapshot in train_dataset:
            assert snapshot.edge_index.shape[0] == 2
            assert snapshot.edge_index.shape[1] == snapshot.edge_attr.shape[0]
            assert snapshot.x.shape == (100, 32)
            assert snapshot.y.shape == (100, )
