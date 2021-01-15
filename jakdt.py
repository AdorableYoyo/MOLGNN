"""Dataset for Graph Isomorphism Network(GIN)
(chen jun): Used for compacted graph kernel dataset in GIN

Data sets include:

MUTAG, COLLAB, IMDBBINARY, IMDBMULTI, NCI1, PROTEINS, PTC, REDDITBINARY, REDDITMULTI5K
https://github.com/weihua916/powerful-gnns/blob/master/dataset.zip
"""

import os
import numpy as np
import collections
import itertools

# FEATURE_NAMES = ['atom_type', 'degree', 'formal_charge', 'hybridization_type',
#    'aromatic', 'chirality_type']
# from .. import backend as F

# from .utils import download, extract_archive, get_download_dir, _get_dgl_url
# from ..graph import DGLGraph
import dgl

_LABEL_AVAILABLE_DATASETS = ["JAK1", "JAK2", "JAK3"]
# Sider has 27 multiple labels
_MultipleLabelDatasets = {
    "Sider": 27,
    "ICLR2020_Sider": 27,
    "ICLR2020_ClinTox": 2,
    "ClinTox_twoLabel": 2,
    "MUV": 17,
    "ToxCast": 617,
    "Tox21": 12,
}
"""
node attributes have 6 digits, which represent
    1. atom type (23, same as GAE),
    2. degree (6, need checking),
    3. formal charge (5, need checking),
    4. hybridization type (7, need checking),
    5. aromatic (bool),
    6. chirality type (4, need checking),
respectively.

Numbers in the parethesis are theoretical max number of types. In our dataset, the
number is very likely to be smaller. Need to be checked.

"""
# ONEHOTENCODING = [0, 1,2,3,5]
ONEHOTENCODING_CODEBOOKS = {}
# when the graph label is not available, we set a dummy label. Although the loss
# contributed by the dummy label will not have an influence to the final performance as
# the classification loss will be multipled by 0, in order to make the loss caculation
# process is running well, we set this dummy label as a valid value. For example, the
# binary crossentropy, we should have lable values from 0 or 1.
DUMMY_LABEL = 0


class GINDataset(object):
    """Datasets for Graph Isomorphism Network (GIN)
    Adapted from https://github.com/weihua916/powerful-gnns/blob/master/dataset.zip.

    The dataset contains the compact format of popular graph kernel datasets, which
    includes: 
    MUTAG, COLLAB, IMDBBINARY, IMDBMULTI, NCI1, PROTEINS, PTC, REDDITBINARY,
    REDDITMULTI5K

    This datset class processes all data sets listed above. For more graph kernel
    datasets, see :class:`TUDataset`

    Paramters
    ---------
    name: str
        dataset name, one of below -
        ('MUTAG', 'COLLAB', \
        'IMDBBINARY', 'IMDBMULTI', \
        'NCI1', 'PROTEINS', 'PTC', \
        'REDDITBINARY', 'REDDITMULTI5K')
    self_loop: boolean
        add self to self edge if true
    degree_as_nlabel: boolean
        take node degree as label and feature if true

    """

    def __init__(
        self,
        name="JAK1",
        self_loop=True,
        datapath="/raid/home/jimmyshen/JAK/clustered/",
        dataset="JAK1_train",
        full_datasets=None,
        featureencoding="onehot",
        debug=False,
    ):
        """Initialize the dataset."""
        self.FEATURE_NAMES = [
            "atom_type",
            "degree",
            "formal_charge",
            "hybridization_type",
            "aromatic",
            "chirality_type",
        ]
        self.ONEHOTENCODING = [0, 1, 2, 3, 5]
        self.debug = debug
        self.featureencoding = featureencoding
        self.full_datasets = full_datasets

        self.name = name  # MUTAG
        self.ds_name = "jak"
        # self.extract_dir = self._download()
        # self.file = self._file_path()

        self.self_loop = self_loop
        self.datapath = datapath
        self.dataset = dataset

        self.graphs = []
        self.labels = []
        self.fingerprints = []

        # relabel
        # self.glabel_dict = {}
        self.glabel_dict = {1: 1, -1: 0, 0: 0}
        self.nlabel_dict = {}
        self.elabel_dict = {}
        self.ndegree_dict = {}

        # global num
        self.N = 0  # total graphs number
        self.n = 0  # total nodes number
        self.m = 0  # total edges number

        # global num of classes
        self.gclasses = 0
        self.nclasses = 0
        self.eclasses = 0
        self.dim_nfeats = 0
        self.fingerprint_dim = 0

        # flags
        self.nattrs_flag = False
        self.nlabels_flag = False
        self.verbosity = False

        # calc all values
        self._load()

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx):
        """Get the i^th sample.

        Paramters
        ---------
        idx : int
            The sample index.

        Returns
        -------
        (dgl.DGLGraph, int)
            The graph and its label.
        """
        return self.graphs[idx], self.labels[idx], self.fingerprints[idx]

    def _get_onehot_encoding_features(self):
        global ONEHOTENCODING_CODEBOOKS
        if ONEHOTENCODING_CODEBOOKS:
            print(
                "ONEHOTENCODING_CODEBOOKS is available already, do not need to"
                "regenerate ONEHOTENCODING_CODEBOOKS"
            )
            print(ONEHOTENCODING_CODEBOOKS)

        else:
            print("regenerating ONEHOTENCODING_CODEBOOKS ...")
            node_attributes_filenames = [
                self.datapath
                + dataset_name
                + "/"
                + dataset_name
                + "_node_attributes.txt"
                for dataset_name in self.full_datasets
            ]
            print(f"node_attributes_filenames, {node_attributes_filenames}")
            node_attributes_all = []
            for i, filename in enumerate(node_attributes_filenames):
                with open(filename, "r") as f:
                    lines = f.readlines()

                this_node_attributes = [
                    [float(l) for l in line.split(",")] for line in lines
                ]
                print(filename, len(this_node_attributes))
                node_attributes_all += this_node_attributes
            print("all files", len(node_attributes_all))
            node_attributes_cnt = {}
            for j, col in enumerate(zip(*node_attributes_all)):
                node_attributes_cnt[self.FEATURE_NAMES[j]] = collections.Counter(col)
            # print(f"node_attributes_cnt, {node_attributes_cnt}")
            for featurename in self.FEATURE_NAMES:
                print(featurename)
                print(sorted(node_attributes_cnt[featurename].items()))
            ONEHOTENCODING_CODEBOOKS.update(
                {
                    feature_name: sorted(node_attributes_cnt[feature_name].keys())
                    for feature_name in self.FEATURE_NAMES
                }
            )
        node_attributes_we_needed_filename = (
            self.datapath + self.dataset + "/" + self.dataset + "_node_attributes.txt"
        )
        with open(node_attributes_we_needed_filename, "r") as f:
            lines = f.readlines()
        node_attributes_we_needed = [
            [float(l) for l in line.split(",")] for line in lines
        ]
        node_attributes_one_hot = []
        # ONEHOTENCODING_CODEBOOKS = {
        #    'atom_type': [0.0, 1.0, 2.0, 3.0, 4.0, 7.0, 8.0, 14.0],
        #    'degree': [1.0, 2.0, 3.0, 4.0],
        #    'formal_charge': [-1.0, 0.0, 1.0],
        #    'hybridization_type': [2.0, 3.0, 4.0],
        #    'aromatic': [0.0, 1.0],
        #    'chirality_type': [0.0, 1.0, 2.0]}
        for row in node_attributes_we_needed:
            this_row = []
            for j, feature_val_before_onehot in enumerate(row):
                if j in self.ONEHOTENCODING:
                    onehot_codebook = ONEHOTENCODING_CODEBOOKS[self.FEATURE_NAMES[j]]
                    onehot_values = [0.0] * len(onehot_codebook)
                    # print(f'one hot code book{onehot_codebook}')
                    # print(f'feat val before one hot {feature_val_before_onehot}')
                    assert feature_val_before_onehot in onehot_codebook
                    onehot_values[
                        onehot_codebook.index(feature_val_before_onehot)
                    ] = 1.0
                    this_row += onehot_values
                else:
                    this_row.append(feature_val_before_onehot)
            node_attributes_one_hot.append(this_row)
        return node_attributes_one_hot

    def _get_graph_raw_data(self):
        A_filename = self.dataset + "_A.txt"
        graph_indicator_filename = self.dataset + "_graph_indicator.txt"
        graph_labels_filename = self.dataset + "_graph_labels.txt"
        graph_fingerprint_filename = self.dataset + "FP_graph_labels.txt"
        node_attributes_filename = self.dataset + "_node_attributes.txt"
        # get A
        with open(self.datapath + self.dataset + "/" + A_filename, "r") as f:
            lines = f.readlines()
        A = [[int(l) for l in line.split(",")] for line in lines]

        # get graph_indicator
        # begins from 1 to N. If we have N graph, the indicator will from 1 to N
        with open(
            self.datapath + self.dataset + "/" + graph_indicator_filename, "r"
        ) as f:
            lines = f.readlines()
        graph_indicator = [int(line) for line in lines]

        # get graph labels
        if self.dataset in _LABEL_AVAILABLE_DATASETS:
            with open(
                self.datapath + self.dataset + "/" + graph_labels_filename, "r"
            ) as f:
                lines = f.readlines()
            graph_labels = [int(line) for line in lines]
            with open(
                self.datapath + self.dataset + "/" + graph_fingerprint_filename, "r"
            ) as f:
                lines = f.readlines()
            graph_fingerprint = [[float(l) for l in line.split(",")] for line in lines]
        else:
            # if no labels are available, set the label as 0.
            graph_labels = [0] * (len(set(graph_indicator)))

        # get node_attributes
        if self.featureencoding == "onehot":
            node_attributes = self._get_onehot_encoding_features()
        else:
            with open(
                self.datapath + self.dataset + "/" + node_attributes_filename, "r"
            ) as f:
                lines = f.readlines()
            node_attributes = [[float(l) for l in line.split(",")] for line in lines]
        assert len(set(graph_indicator)) == len(graph_labels)
        assert len(set(graph_indicator)) == len(graph_fingerprint)
        assert len(graph_indicator) == len(node_attributes)
        return A, graph_indicator, graph_labels, node_attributes, graph_fingerprint

    def _get_nodes_per_graph(self, graph_indicator):
        return [len(list(g)) for i, g in itertools.groupby(graph_indicator)]

    def _load(self):
        """ Loads input dataset from the following files
        JAK1_train_A.txt  
        JAK1_train_edge_attributes.txt  
        JAK1_train_graph_indicator.txt  
        JAK1_train_graph_labels.txt  
        JAK1_train_node_attributes.txt
        """
        print("loading data...")
        raw_label_dict = collections.Counter()
        new_label_dict = collections.Counter()
        (
            A,
            graph_indicator,
            graph_labels,
            node_attributes,
            graph_fingerprint,
        ) = self._get_graph_raw_data()
        self.fingerprint_dim = len(graph_fingerprint[0])
        print("A unique node idx", len({edge[0] for edge in A}))
        self.N = len(graph_labels)
        nodes_per_graph = self._get_nodes_per_graph(graph_indicator)
        if self.debug:
            print(f"(nodes_per_graph[:10]: {nodes_per_graph[:10]}")
        assert len(nodes_per_graph) == len(graph_labels)
        TOTAL_NUM_NODES = sum(nodes_per_graph)
        self.n = TOTAL_NUM_NODES
        assert TOTAL_NUM_NODES == len(node_attributes)
        assert TOTAL_NUM_NODES == max(max(edge) for edge in A)
        begin_node_idx, end_node_idx = 1, 0
        raw_large_edges_idx = 0
        for graph_id, raw_label in enumerate(graph_labels):
            if self.debug:
                print("*" * 20 + str(graph_id))
            m_edges = 0
            # -1	inactive	0	moderate	1	active
            raw_label_dict[raw_label] += 1
            assert raw_label in [-1, 0, 1]
            new_label = self.glabel_dict[raw_label]
            assert new_label in [0, 1]
            new_label_dict[new_label] += 1

            self.labels.append(new_label)
            nodes_of_this_g = nodes_per_graph[graph_id]
            if self.debug:
                print(f"nodes_of_this_g, {nodes_of_this_g}")
            g = dgl.DGLGraph()
            g.add_nodes(nodes_of_this_g)
            end_node_idx = begin_node_idx + nodes_of_this_g - 1
            # change begin_node_idx from base 1 to base 0
            features = np.array(node_attributes[begin_node_idx - 1 : end_node_idx])
            fingerprint = np.array(graph_fingerprint[graph_id], dtype=float)
            self.fingerprints.append(fingerprint)
            if self.debug:
                print("features", features.shape, features)
            g.ndata["attr"] = features
            # g.ndata['fingerprint'] = fingerprint
            while True:
                if raw_large_edges_idx >= len(A):
                    break
                old_u, old_v = A[raw_large_edges_idx]
                if old_u > end_node_idx:
                    #                     assert (
                    #                         old_u == end_node_idx + 1
                    #                     ), "old_u, {}, old_v {}, end_node_idx{}".format(
                    #                         old_u, old_v, end_node_idx
                    #                     )
                    begin_node_idx = end_node_idx + 1
                    break
                assert begin_node_idx <= old_v <= end_node_idx
                # if we begin from 1 to 10.
                # the following will change it to 0 based index
                new_u, new_v = old_u - begin_node_idx, old_v - begin_node_idx
                if self.debug:
                    print("edge:", old_u, old_v, new_u, new_v)
                g.add_edge(new_u, new_v)
                m_edges += 1
                raw_large_edges_idx += 1
            if self.self_loop:
                for k in range(nodes_of_this_g):
                    m_edges += 1
                    g.add_edge(k, k)
            self.graphs.append(g)
            self.m += m_edges

        # after load, get the #classes and #dim
        self.gclasses = len(set(self.glabel_dict.values()))
        self.nclasses = len(self.nlabel_dict)
        self.eclasses = len(self.elabel_dict)
        self.dim_nfeats = len(self.graphs[0].ndata["attr"][0])

        print("Done.")
        print(
            """
            -------- Data Statistics --------'
            #Graphs: %d
            #Graph Classes: %d
            #Nodes: %d
            #Node Classes: %d
            #Node Features Dim: %d
            #Edges: %d
            #Edge Classes: %d
            Avg. of #Nodes: %.2f
            Avg. of #Edges: %.2f
            Graph Relabeled: %s
            Node Relabeled: %s
            Raw Label Relabeled: %s
            New Label Relabeled: %s   \n """
            % (
                self.N,
                self.gclasses,
                self.n,
                self.nclasses,
                self.dim_nfeats,
                self.m,
                self.eclasses,
                self.n / self.N,
                self.m / self.N,
                self.glabel_dict,
                self.nlabel_dict,
                raw_label_dict,
                new_label_dict,
            )
        )


class DeepChemDataset(object):
    """Datasets for DeepChem
    DeepChem is already loaded by using PyTorch Geometric 
    the dataset from PyTorch Geometric is named dataset_pg. It has the format as
    [graph0, grpah1, graph2, graph3 ..] each graph is a Data() instance and the class
    Data is defined here:
    the fingerprint data is saved in dataset_fingerprint_pg. It has the format as
    [graph0, graph1, graph2, graph3 ...]. The label for the dataset_fingerprint_pg is
    the fingerprint.
    https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/data/data.py
    for each graph, you can have some outputs like this:
    Data(edge_index=[2, 30], x=[15, 6], y=[1])
    data.x: it is the feature tensor
tensor([[1., 1., 0., 3., 0., 0.],
        [0., 3., 0., 3., 0., 0.],
        [1., 1., 0., 3., 0., 0.],
        [1., 2., 0., 3., 0., 0.],
        [0., 3., 0., 3., 0., 0.],
        [2., 1., 0., 3., 0., 0.],
        [0., 3., 0., 3., 1., 0.],
        [1., 2., 0., 3., 1., 0.],
        [0., 3., 0., 3., 1., 0.],
        [7., 1., 0., 4., 0., 0.],
        [0., 3., 0., 3., 1., 0.],
        [1., 1., 0., 3., 0., 0.],
        [1., 2., 0., 3., 1., 0.],
        [0., 3., 0., 3., 1., 0.],
        [1., 1., 0., 3., 0., 0.]])
        data.edge_index
tensor([[ 0,  1,  1,  1,  2,  3,  3,  4,  4,  4,  5,  6,  6,  6,  7,  7,  8,  8,
          8,  9, 10, 10, 10, 11, 12, 12, 13, 13, 13, 14],
        [ 1,  0,  2,  3,  1,  1,  4,  3,  5,  6,  4,  4,  7, 13,  6,  8,  7,  9,
         10,  8,  8, 11, 12, 10, 10, 13,  6, 12, 14, 13]])

    Paramters
    ---------
    name: str
    dataset_pg: dataset from PyTorch Geometric
    self_loop: boolean
        add self to self edge if true
    """

    def __init__(
        self,
        dataset_pg,
        dataset_fingerprint_pg,
        self_loop=True,
        featureencoding="onehot",
        datapath="/raid/home/jimmyshen/JAK/clustered/",
        full_datasets=None,
        debug=False,
        main_dataset=None,
    ):  # main_dataset is the dataset that we are used to train the graph classifier
        """Initialize the dataset."""
        self.graph_labels_available = True
        if dataset_pg is None:
            print(
                "graph labels are not available for this dataset, set the"
                " self.dataset_pg the same as dataset_fingerprint_pg to keep the old "
                "framework working  "
            )
            self.graph_labels_available = False
            # warning, here by setting the dataset_pg to dataset_fingerprint_pg is only
            # trying to make the old codebase working.
            self.dataset_pg = dataset_fingerprint_pg
        else:
            self.dataset_pg = dataset_pg
        self.main_dataset = main_dataset
        if "ICLR20" not in self.main_dataset:
            self.FEATURE_NAMES = [
                "atom_type",
                "degree",
                "formal_charge",
                "hybridization_type",
                "aromatic",
                "chirality_type",
            ]
            self.ONEHOTENCODING = [0, 1, 2, 3, 5]
        else:
            self.FEATURE_NAMES = ["atom_type", "chirality_type"]
            self.ONEHOTENCODING = [0, 1]
        print(self.FEATURE_NAMES, self.FEATURE_NAMES)
        print("self.ONEHOTENCODING", self.ONEHOTENCODING)
        self.dataset_fingerprint_pg = dataset_fingerprint_pg
        self.datapath = datapath
        self.full_datasets = full_datasets
        self.debug = debug
        self.featureencoding = featureencoding
        self.self_loop = self_loop

        self.graphs = []
        self.labels = []
        self.fingerprints = []

        # relabel
        # self.glabel_dict = {}
        self.glabel_dict = {1: 1, 0: 0}
        self.nlabel_dict = {}
        self.elabel_dict = {}
        self.ndegree_dict = {}

        # global num
        self.N = 0  # total graphs number
        self.n = 0  # total nodes number
        self.m = 0  # total edges number

        # global num of classes
        self.gclasses = 2
        self.nclasses = 0
        self.eclasses = 0
        self.dim_nfeats = 0
        self.fingerprint_dim = 0

        # flags
        self.nattrs_flag = False
        self.nlabels_flag = False
        self.verbosity = False

        # calc all values
        self._parse_dataset_pg()

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __getitem__(self, idx):
        """Get the i^th sample.
        Paramters
        ---------
        idx : int
            The sample index.
        Returns
        -------
        (dgl.DGLGraph, int)
            The graph and its label.
        """
        return self.graphs[idx], self.labels[idx], self.fingerprints[idx]

    def _get_ONEHOTENCODING_CODEBOOKS(self):
        features_all = [data.x.numpy() for data in self.dataset_pg]
        features = np.vstack(features_all)
        print(
            "the node feature size if we put all graphs node features together",
            features.shape,
        )
        features = features.tolist()
        print("before adding extra dataset, the size of featues is", len(features))
        node_attributes_filenames = [
            self.datapath + dataset_name + "/" + dataset_name + "_node_attributes.txt"
            for dataset_name in self.full_datasets
        ]
        print(f"node_attributes_filenames, {node_attributes_filenames}")
        node_attributes_all = []
        for i, filename in enumerate(node_attributes_filenames):
            with open(filename, "r") as f:
                lines = f.readlines()
                this_node_attributes = [
                    [float(l) for l in line.split(",")] for line in lines
                ]
                print(filename, len(this_node_attributes))
                node_attributes_all += this_node_attributes
        print("all files", len(node_attributes_all))
        features += node_attributes_all
        print("after adding extra dataset, the size of featues is", len(features))
        node_attributes_cnt = {}
        for j, col in enumerate(zip(*features)):
            node_attributes_cnt[self.FEATURE_NAMES[j]] = collections.Counter(col)
        # print(f"node_attributes_cnt, {node_attributes_cnt}")
        for featurename in self.FEATURE_NAMES:
            print(featurename)
            # ICLR20_* dataset may contain only two features.
            if featurename in node_attributes_cnt:
                print(sorted(node_attributes_cnt[featurename].items()))
        ONEHOTENCODING_CODEBOOKS.update(
            {
                feature_name: sorted(node_attributes_cnt[feature_name].keys())
                for feature_name in self.FEATURE_NAMES
            }
        )
        print("generated ONEHOTENCODING_CODEBOOKS")
        print(ONEHOTENCODING_CODEBOOKS)

    def _get_onehot_encoding_features(self, raw_feature):
        feature_one_hot = []
        for row in raw_feature.tolist():
            this_row = []
            for j, feature_val_before_onehot in enumerate(row):
                if j in self.ONEHOTENCODING:
                    onehot_codebook = ONEHOTENCODING_CODEBOOKS[self.FEATURE_NAMES[j]]
                    onehot_values = [0.0] * len(onehot_codebook)

                    assert feature_val_before_onehot in onehot_codebook
                    onehot_values[
                        onehot_codebook.index(feature_val_before_onehot)
                    ] = 1.0
                    this_row += onehot_values
                else:
                    this_row.append(feature_val_before_onehot)
            feature_one_hot.append(this_row)
        return np.array(feature_one_hot)

    def _parse_dataset_pg(self):
        self.N = len(self.dataset_pg)
        if self.featureencoding == "onehot":
            self._get_ONEHOTENCODING_CODEBOOKS()
            if self.debug:
                print("ONEHOTENCODING_CODEBOOKS", ONEHOTENCODING_CODEBOOKS)
        # assert len(self.dataset_pg) == len(self.dataset_fingerprint_pg), "ataset_pg
        # should have the same size as the dataset_fingerprint_pg, we get dataset_pg
        # size of {} dataset_fingerprint_pg size of {}".format(len(self.dataset_pg),
        # len(self.dataset_fingerprint_pg))
        datasample_counter = 0
        for data, data_fingerprint in zip(self.dataset_pg, self.dataset_fingerprint_pg):
            # print("data", data)
            # print(self.dataset_pg[0])
            # print(self.dataset_fingerprint_pg[0])
            # print(self.dataset_fingerprint_pg[1])
            # print("data_fingerprint", data_fingerprint)
            edge_index, feature = data.edge_index, data.x.numpy()
            if self.graph_labels_available:
                if self.main_dataset in _MultipleLabelDatasets:
                    # print("data.y before to list", data.y)
                    label = data.y[0].tolist()
                    # print("label", label)
                    # print("data.y after to list", data.y)
                    self.gclasses = _MultipleLabelDatasets[self.main_dataset]
                    assert (
                        len(label) == _MultipleLabelDatasets[self.main_dataset]
                    ), "label dimension should match, len of label is {}".format(
                        len(label)
                    )
                else:
                    label = int(data.y)
            else:
                if self.main_dataset in _MultipleLabelDatasets:
                    label = [DUMMY_LABEL] * _MultipleLabelDatasets[self.main_dataset]
                else:
                    label = DUMMY_LABEL
            edge_index_from_fp_data, feature_from_fingerprint_data, fingerprint = (
                data_fingerprint.edge_index,
                data_fingerprint.x.numpy(),
                data_fingerprint.y.numpy(),
            )
            fingerprint = np.array(fingerprint, dtype=float)
            # the original fingerprint has the shape of (a, 740), change it to shape of
            # (740,)
            fingerprint = fingerprint.ravel()
            self.fingerprint_dim = fingerprint.shape[0]
            self.fingerprints.append(fingerprint)
            # np.savetxt(str(datasample_counter)+"feature_notfrom_fp.txt",feature)
            # np.savetxt(
            #   str(datasample_counter)+"feature_from_fp.txt",
            #   feature_from_fingerprint_data)
            assert (
                feature.shape[0] == feature_from_fingerprint_data.shape[0]
            ), "the feature shpae 0 (number of nodes in the graph) from those two \
            datasets should be eqaul. sample index is {}".format(
                datasample_counter
            )
            assert np.array_equal(
                edge_index, edge_index_from_fp_data
            ), "the feature from those two dataset should be eqaul"
            assert np.array_equal(
                feature, feature_from_fingerprint_data
            ), "the feature from those two dataset should be eqaul {}".format(
                datasample_counter
            )

            datasample_counter += 1
            # print("label", label)
            # assert label in [0 ,1, DUMMY_LABEL]
            self.labels.append(label)
            u, v = edge_index
            if self.featureencoding == "onehot":
                if self.debug:
                    print("before onehot encoding", feature)
                feature = self._get_onehot_encoding_features(feature)
                if self.debug:
                    print("after onehot encoding", feature)
            nodes_of_this_g = data.x.shape[0]
            self.n += nodes_of_this_g
            edges = {(u, v) for u, v in zip(u.tolist(), v.tolist())}
            if self.self_loop:
                for k in range(nodes_of_this_g):
                    edges.add((k, k))
            g = dgl.DGLGraph()
            g.add_nodes(nodes_of_this_g)
            g.ndata["attr"] = feature
            for u, v in edges:
                g.add_edge(u, v)
            self.graphs.append(g)
            self.m += len(edges)
        # print("label statistics: ", collections.Counter(self.labels))
        self.dim_nfeats = len(self.graphs[0].ndata["attr"][0])
        print("Done.")
        print(
            """
            -------- Data Statistics --------'
            #Graphs: %d
            #Graph Classes: %d
            #Nodes: %d
            #Node Classes: %d
            #Node Features Dim: %d
            #Edges: %d
            #Edge Classes: %d
            #Fingerprint dim: %d
            Avg. of #Nodes: %.2f
            Avg. of #Edges: %.2f \n """
            % (
                self.N,
                self.gclasses,
                self.n,
                self.nclasses,
                self.dim_nfeats,
                self.m,
                self.eclasses,
                self.fingerprint_dim,
                self.n / self.N,
                self.m / self.N,
            )
        )
