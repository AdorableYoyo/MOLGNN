# load DeepChem dataset


import torch
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
import collections
import dgl
import numpy as np

def show_topk(dataset, k = 10):
    features = []
    for data in dataset[:k]:
        print(data)
        print(data.x)
        features.append(data.x.numpy())
        print(type(data.edge_index))
        u, v = data.edge_index
        print(u, v)
        nodes_of_this_g = data.x.shape[0]
        print('nodes_of_this_g', nodes_of_this_g)
        g = dgl.DGLGraph()
        g.add_nodes(nodes_of_this_g)
        edges = {(u, v) for u, v in zip(u.tolist(), v.tolist())}
        for u, v in edges:
            g.add_edge(u, v)
        g.ndata['attr'] = data.x
        print(g)
    print("combined features")
    features = np.vstack(features)
    print(features.shape)
    print(features)
    ys = [int(data.y) for data in dataset]
    print(collections.Counter(ys))

class DeepChemDatasetPG(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(DeepChemDatasetPG, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass
    
    def process(self):
        pass
#dataset = DeepChemDatasetPG('/raid/home/jimmyshen/DeepChem/BBBP/')
#print(len(dataset))
#show_topk(dataset, 10)
