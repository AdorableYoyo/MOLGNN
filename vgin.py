"""
How Powerful are Graph Neural Networks
https://arxiv.org/abs/1810.00826
https://openreview.net/forum?id=ryGs6iA5Km
Author's implementation: https://github.com/weihua916/powerful-gnns
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

DEBUG = False

class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)
        self.debug = False

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction

        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction

        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.debug = DEBUG

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
        print(f"self.linear_or_not{self.linear_or_not}")

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
                if self.debug:
                    print(f"i {i}")
                    print(f"h.size() {h.size()}")
                    print(f"h {h}")
            return self.linears[-1](h)


class GIN(nn.Module):
    """GIN model"""
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type):
        """model parameters setting

        Paramters
        ---------
        num_layers: int
            The number of linear layers in the neural network
        num_mlp_layers: int
            The number of linear layers in mlps
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        final_dropout: float
            dropout ratio on the final linear layer
        learn_eps: boolean
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: str
            how to aggregate neighbors (sum, mean, or max)
        graph_pooling_type: str
            how to aggregate entire nodes in a graph (sum, mean or max)

        """
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps
        self.debug = DEBUG

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, h):
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]
        if self.debug:
            print(f"g, {g}")
            print(f"h.size {h.size()}")
            print(f"h {h}")
            print(f"num_layers, {self.num_layers}")
        for i in range(self.num_layers - 1):
            if self.debug:
                print(f"i {i}")
                print(f"h.size {h.size()}")
                print('h begining', h)
            h = self.ginlayers[i](g, h)
            if self.debug:
                print(f"h.size {h.size()}")
                print('h after gin', h)
            #h = self.batch_norms[i](h)
            #print('h after gin', h)
            h = F.relu(h)
            if self.debug:
                print(f"h.size {h.size()}")
                print('h after relu', h)
            hidden_rep.append(h)
        if self.debug:
            print(f"hidden_rep {hidden_rep}")
        score_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            if self.debug:
                print(f"h.size {h.size()}")
                print(f"i, h in hidden_rep {h}")
            pooled_h = self.pool(g, h)
            if self.debug:
                print(f"h.size {h.size()}")
                print(f"pooled_h {pooled_h}")

            #score_over_layer += self.drop(self.linears_prediction[i](pooled_h))
            this_layer= self.linears_prediction[i](pooled_h)
            if self.debug:
                print(f"this_layer, {this_layer.size()}")
                print(f"this_layer, {this_layer}")
            score_over_layer += self.linears_prediction[i](pooled_h)
            if self.debug:
                print(f"score_over_layer {score_over_layer.size()}")
                print(f"score_over_layer {score_over_layer}")

        return score_over_layer
    
# Degine a GIN and VGAE model to train the GIN and VGAE at the same time.
class GIN_VGAE(nn.Module):
    """ 
        GIN_VGAE model for the layer name, we keep on using the gae 
        to make it consistent with the gae model
    """
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,fingerprint_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type):
        """model parameters setting
        Paramters
        ---------
        num_layers: int
            The number of linear layers in the neural network
        num_mlp_layers: int
            The number of linear layers in mlps
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        fingerprint_dim: int
             The dimensionality of the fingerprint
        output_dim: int
            The number of classes for prediction
        final_dropout: float
            dropout ratio on the final linear layer
        learn_eps: boolean
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: str
            how to aggregate neighbors (sum, mean, or max)
        graph_pooling_type: str
            how to aggregate entire nodes in a graph (sum, mean or max)
        """
        super(GIN_VGAE, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps
        self.debug = False
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim)

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction_gae = torch.nn.ModuleList()
        self.linears_prediction_classification = torch.nn.ModuleList()
        self.linears_prediction_fingerprint = torch.nn.ModuleList()
        # for gae the embedding dimension is equal to the hidden dim
        # for gin, the output dimension is the output dimension, such as if the classification category is 3, 
        # we will have output dimension as 3.
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction_gae.append(
                    nn.Linear(input_dim,2*hidden_dim))
                self.linears_prediction_classification.append(
                    nn.Linear(input_dim, output_dim))
                self.linears_prediction_fingerprint.append(
                    nn.Linear(input_dim, hidden_dim))
            else:
                self.linears_prediction_gae.append(
                    nn.Linear(hidden_dim, 2*hidden_dim))
                self.linears_prediction_classification.append(
                    nn.Linear(hidden_dim, output_dim))
                self.linears_prediction_fingerprint.append(
                    nn.Linear(hidden_dim, hidden_dim))

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError
        self.decoder = InnerProductDecoder(activation=lambda x:x)
        self.fingerprint_decoder = FingerprintDecoder(hidden_dim, fingerprint_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, g, h):
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]
        if self.debug:
            print(f"g, {g}")
            print(f"h.size {h.size()}")
            print(f"h {h}")
            print(f"num_layers, {self.num_layers}")
        for i in range(self.num_layers - 1):
            if self.debug:
                print(f"i {i}")
                print(f"h.size {h.size()}")
                print('h begining', h)
            h = self.ginlayers[i](g, h)
            if self.debug:
                print(f"h.size {h.size()}")
                print('h after gin', h)
            #h = self.batch_norms[i](h)
            #print('h after gin', h)
            h = F.relu(h)
            if self.debug:
                print(f"h.size {h.size()}")
                print('h after relu', h)
            hidden_rep.append(h)
        if self.debug:
            print(f"hidden_rep {hidden_rep}")
        score_over_layer_gae = 0
        score_over_layer_classification = 0
        score_over_layer_fingerprint = 0
        # perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            if self.debug:
                print(f"h.size {h.size()}")
                print(f"i, h in hidden_rep {h}")
            pooled_h = self.pool(g, h)
            if self.debug:
                print(f"h.size {h.size()}")
                print(f"pooled_h {pooled_h}")

            #score_over_layer += self.drop(self.linears_prediction[i](pooled_h))
            this_layer_gae= self.linears_prediction_gae[i](h)
            this_layer_classification= self.linears_prediction_classification[i](pooled_h)
            this_layer_fingerprint= self.linears_prediction_fingerprint[i](pooled_h)
            print(f"this_layer_classification, {this_layer_classification.size()}")
            print(f"this_layer_classification, {this_layer_classification}")
            if self.debug:
                print(f"this_layer, {this_layer_gae.size()}")
                print(f"this_layer, {this_layer_gae}")
            score_over_layer_gae += this_layer_gae
            score_over_layer_classification += this_layer_classification
            score_over_layer_fingerprint += this_layer_fingerprint
            print(f"score_over_layer_classification, {score_over_layer_classification.size()}")
            print(f"score_over_layer_classification, {score_over_layer_classification}")
            if self.debug:
                print(f"score_over_layer {score_over_layer_gae.size()}")
                print(f"score_over_layer {score_over_layer_gae}")
            fingerprint_rec = self.fingerprint_decoder(score_over_layer_fingerprint)
            #fingerprint_rec = self.fingerprint_decoder(score_over_layer_gae)
            mu, logvar = (score_over_layer_gae[...,:self.hidden_dim],
                          score_over_layer_gae[...,self.hidden_dim:])
            z = self.reparameterize(mu, logvar)
            adj_rec = self.decoder(z)

        return adj_rec, mu, logvar, score_over_layer_classification, fingerprint_rec
         
    
class InnerProductDecoder(nn.Module):
    def __init__(self, activation=torch.sigmoid, dropout=0.1):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.activation = activation

    def forward(self, z):
        z = F.dropout(z, self.dropout)
        adj = self.activation(torch.mm(z, z.t()))
        return adj
    
class FingerprintDecoder(nn.Module):
    def __init__(self, n_in, n_out, dropout=0.1):
        # n_in is the same as the hidden representation of the VGAE.
        super(FingerprintDecoder, self).__init__()
        if n_out > n_in:
            n_hidden = n_out // 2
        else:
            n_hidden = n_in // 2
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self.dropout = dropout

    def forward(self, x):
        # input is hidden representation which has the shape of [None, 1, Hidden] 
        # --> [None, 1, this hidden] --> [None, 1, output dim] 
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(x)
        # this is different to Yang's network as we are directly processing based on the N*hidden output of VGAE
        # x = torch.mean(x, -2)
        return x
