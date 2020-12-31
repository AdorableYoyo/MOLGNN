"""
PyTorch compatible dataloader
"""


import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
import dgl
import math

# default collate function
def collate(samples):
    # The input `samples` is a list of pairs (graph, label).
    graphs, labels, fingerprints = map(list, zip(*samples))
    for g in graphs:
        # deal with node feats
        for key in g.node_attr_schemes().keys():
            g.ndata[key] = g.ndata[key].float()
        # no edge feats
    batched_graph = dgl.batch(graphs)
    labels = torch.tensor(labels)
    fingerprints =  torch.tensor(fingerprints)
    return batched_graph, labels, fingerprints


class GraphDataLoader():
    def __init__(self,
                 dataset,
                 batch_size,
                 device,
                 collate_fn=collate,
                 shuffle=True):
        self.kwargs = {'pin_memory': True} if 'cuda' in device.type else {}
        self.data_loader = DataLoader(dataset,
                                      batch_size=batch_size, 
                                      collate_fn=collate_fn,
                                      shuffle=shuffle,
                                      **self.kwargs)
    def get_data_loader(self):
        return self.data_loader

"""
A DataLoader which support the random splitting. Hence, it can output two dataset: train and test
"""
class GraphDataLoaderSplit():
    def __init__(self,
                 dataset,
                 stage,
                 batch_size,
                 device,
                 collate_fn=collate,
                 split_ratio=0.9,
                 shuffle=True,
                 seed=0,
                 data_splitting_method="random_split",
                 scaffold_split_idx_path=None,
                 main_dataset=None,
                 USING_CORRECT_scaffold_split_implementation=True): # see follwing comments about this dangerous argument
        """
        C. Detail explanation of the problem of the ICLR 2020 paper's wrong implementation of scattering splitting.
        The problem of their implementation for the scatter splitting is:
        When the cluster has the same size, it is further sorted by the label of the first element of that cluster.
        By doing this, we may have validation or test split (especially for the test dataset) that has zero samples
        of some categories. 
        We discovered this error when we applied the same implementation on the JAK1/2/3 dataset.

        One example (maybe not good, just show why their implementation got an error) used to explain the problem of
        their original implementation is:supposed we already have the train split filled and we need 8 samples for
        validation and 4 for testing. Here are some candidates: 

        clusterA: [(sample1, label=0),  (sample 3, label = 1), ]
        clusterB: [(sample2, label=0),  (sample 4, label = 1), ]
        clusterC: [(sample9, label=1),  (sample 5, label = 1), ]
        clusterD: [(sample10, label=1),  (sample 8, label = 1), ]
        clusterE: [(sample11, label=0),  (sample 9, label = 1), ]
        clusterF: [(sample12, label=0),  (sample 13, label = 1), ]

        Their implementation is sort by the length of the cluster and then by the label of the first label, 
        so the splitting will be: 
        
        Validation:
        clusterA: [(sample1, label=0),  (sample 3, label = 1), ]
        clusterB: [(sample2, label=0),  (sample 4, label = 1), ]
        clusterE: [(sample11, label=0),  (sample 9, label = 1), ]
        clusterF: [(sample12, label=0),  (sample 13, label = 1), ]
        
        Test:
        clusterC: [(sample9, label=1),  (sample 5, label = 1), ]
        clusterD: [(sample10, label=1),  (sample 8, label = 1), ]

        We can see that the test has only samples in category 1 and no samples in category 0.
        
        Why we still use the wrong implementation?
        In order to have a fair comparison, we use the wrong implementation only for fair comparison. 
        Our main results are reported based on the CORRECT implementation.
        """
        self.seed = seed
        self.stage = stage
        self.scaffold_split_idx_path = scaffold_split_idx_path
        self.data_splitting_method = data_splitting_method
        self.main_dataset = main_dataset
        self.USING_CORRECT_scaffold_split_implementation = USING_CORRECT_scaffold_split_implementation
        if not self.USING_CORRECT_scaffold_split_implementation:
            assert self.data_splitting_method == "scaffold_split", "Only scaffold split can use this WRONG implementation"
            #assert "ICLR2020_" in self.main_dataset, "We label the ICLR approach with the prefix of ICLR_2020"
        self.kwargs = {'pin_memory': True} if 'cuda' in device.type else {}
        labels = [l for _, l, _ in dataset]
        self._scaffold_split_ratio = [0.8, 0.1, 0.1]
        if self.data_splitting_method == "scaffold_split":
            train_idx, valid_idx, test_idx = self._get_scaffold_split()
            # We may get out of index error, the 3 line code below is debug this kind of error
            print("Possible invalid train_idx", [x for x in train_idx if x>=len(labels)])
            print("Possible invalid valid_idx", [x for x in valid_idx if x>=len(labels)])
            print("Possible invalid test_idx", [x for x in test_idx if x>=len(labels)])
            train_idx = [x for x in train_idx if x<len(labels)]
            valid_idx = [x for x in valid_idx if x<len(labels)]
            test_idx = [x for x in test_idx if x<len(labels)]
        else:
            assert self.stage != 'validation', "This version code for the random split only suporting test stage"
            train_idx, test_idx = self._split_rand(labels, split_ratio)
            print("random_split is used")
            print("Number of train sample: ", len(train_idx))
            print("Number of test sample: ", len(test_idx))
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        self.train_loader = DataLoader(dataset,
                                      sampler=train_sampler,
                                      batch_size=batch_size, 
                                      collate_fn=collate_fn,
                                      shuffle=False,
                                      **self.kwargs)
        self.test_loader = DataLoader(dataset,
                                      sampler=test_sampler,
                                      batch_size=batch_size, 
                                      collate_fn=collate_fn,
                                      shuffle=False,
                                      **self.kwargs)
        self.valid_loader = None
        if self.data_splitting_method == "scaffold_split":
            print("scaffold_split is used")
            print("length of the labels: ", len(labels))
            print("Number of train sample: ", len(train_idx))
            print("Number of valid sample: ", len(valid_idx))
            print("Number of test sample: ", len(test_idx))
            if self.stage == 'validation':
                valid_sampler = SubsetRandomSampler(valid_idx)
                self.valid_loader = DataLoader(dataset,
                                      sampler=valid_sampler,
                                      batch_size=batch_size, 
                                      collate_fn=collate_fn,
                                      shuffle=False,
                                      **self.kwargs)

    def _read_idx(self, filename):
        """
        A utility function to read txt file
        """
        with open(filename, "r") as f:
            lines = f.readlines()
        return [int(line) for line in lines]
    
    def _get_scaffold_split(self):
        # the original code may make we have less train samples and more validation and test samples. The code here
        # can make the ratio correct.
        assert self.scaffold_split_idx_path is not None, "scaffold_split_idx_path can not be None"
        assert self.data_splitting_method == "scaffold_split", "this function can only be used by scaffold split"
        #train_idx, valid_idx, test_idx = [self._read_idx(self.scaffold_split_idx_path+f+"_idx.txt") 
                #for f in ["new_train", "new_val", "new_test"]]
        if not self.USING_CORRECT_scaffold_split_implementation:
            # old one is the wrongly implemented one. The new one has the prefix of new and the old one has NO prefix
            WRONG_split_id_filenames = ["train", "val", "test"]
            print("WARNING, the old scaffold_split idx are used, the files names are", WRONG_split_id_filenames)
            train_idx, valid_idx, test_idx = [self._read_idx(self.scaffold_split_idx_path+f+"_idx.txt") 
                                              for f in WRONG_split_id_filenames]
        if self.stage == 'test':
            train_idx = train_idx + valid_idx
            valid_idx = []
        return train_idx, valid_idx, test_idx

    def _split_rand(self, labels, split_ratio):
        num_entries = len(labels)
        indices = list(range(num_entries))
        np.random.seed(self.seed)
        np.random.shuffle(indices)
        split = int(math.floor(split_ratio * num_entries))
        train_idx, valid_idx = indices[:split], indices[split:]

        print(
            "train_set : test_set = %d : %d",
            len(train_idx), len(valid_idx))

        return train_idx, valid_idx
    
    def get_data_loader(self):
        #return self.train_loader, self.valid_loader, self.test_loader
        return self.train_loader, self.valid_loader, self.test_loader
