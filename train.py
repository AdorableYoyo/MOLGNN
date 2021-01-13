from tqdm import tqdm
import sys, os
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim
from optimizer import loss_function as variational_loss_function
from torch.nn.functional import binary_cross_entropy_with_logits as BCELoss
from parser_MoLGNN import Parser
from jakdt import  _MultipleLabelDatasets

def train(args,
          net, 
          trainloader, 
          optimizer, 
          criterion_gae, 
          criterion_classification, 
          criterion_fingerprint,
          epoch,
          gae_weight_rt=0,
          classification_weight_rt=1.0,
          fingerprint_weight_rt=1.0):
    if trainloader is None:
        print("Found an empty loader, this might be the empty validloader")
        return
    net.train()
    
    running_loss_gae_original = 0
    running_loss_classification_original = 0
    running_loss_fingerprint_original = 0
    
    running_loss_gae_weighted = 0
    running_loss_classification_weighted = 0
    running_loss_fingerprint_weighted = 0
    
    running_loss = 0
    
    total_iters = len(trainloader)
    # setup the offset to avoid the overlap with mouse cursor
    bar = tqdm(range(total_iters), unit='batch', position=2, file=sys.stdout)

    for pos, (graphs, labels, fingerprints) in zip(bar, trainloader):
        # batch graphs will be shipped to device in forward part of model
        if  args.dataset in _MultipleLabelDatasets:
            labels = labels.double()
        labels = labels.to(args.device)
        fingerprints = fingerprints.to(args.device)
        feat = graphs.ndata['attr'].to(args.device)
        #fingerprint_gt = graphs.ndata['fingerprint'].to(args.device)
        fingerprint_gt = fingerprints
        adj = graphs.adjacency_matrix().to_dense()
        adj_np = adj.numpy()
        adj = sp.csr_matrix(adj_np)
        # update device here
        adj_label = torch.FloatTensor(adj.toarray()).to(args.device)
        
        pos_weight = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()]).to(args.device)
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        adj_rec, mu, logvar,score_over_layer_classification, fingerprint_rec = net(graphs, feat)
        print("fingerprint_gt.shape, fingerprint_rec.shape", fingerprint_gt.shape, fingerprint_rec.shape)
        #norm*loss_function(adj_logits, adj_label, pos_weight=pos_weight)
        # this is loss vgae
        loss_gae = variational_loss_function(preds=adj_rec, labels=adj_label,
                                 mu=mu, logvar=logvar, n_nodes=adj.shape[0],
                                 norm=norm, pos_weight=pos_weight, vgae_loss=True)
        #loss_gae = norm*criterion_gae(adj_rec, adj_label, pos_weight=pos_weight)
        #print("score_over_layer_classification.size(), labels.size()", score_over_layer_classification.size(), labels.size())
        loss_classification = criterion_classification(score_over_layer_classification, labels)
        loss_fingerprint = criterion_fingerprint(fingerprint_rec, fingerprint_gt)/740
        running_loss_gae_original += loss_gae.item()
        running_loss_classification_original += loss_classification.item()
        running_loss_fingerprint_original += loss_fingerprint.item()
        # rt is short for real time
        loss_gae_weighted = gae_weight_rt*loss_gae
        loss_classification_weighted = classification_weight_rt*loss_classification
        loss_fingerprint_weighted = fingerprint_weight_rt*loss_fingerprint
        
        running_loss_gae_weighted += loss_gae_weighted.item()
        running_loss_classification_weighted += loss_classification_weighted.item()
        running_loss_fingerprint_weighted += loss_fingerprint_weighted.item()
        
        loss = loss_gae_weighted+loss_classification_weighted + loss_fingerprint_weighted
        running_loss += loss.item()

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # report
        bar.set_description('epoch-{}'.format(epoch))
    bar.close()
    # the final batch will be aligned
    running_loss = running_loss / total_iters
    
    # new added items
    running_loss_gae_original /= total_iters
    running_loss_classification_original /= total_iters
    running_loss_gae_weighted /= total_iters
    running_loss_classification_weighted /= total_iters
    running_loss_classification_weighted /= total_iters

    return (running_loss,
            running_loss_gae_original,
            running_loss_classification_original,
            running_loss_fingerprint_original,
            running_loss_gae_weighted,
            running_loss_classification_weighted,
            running_loss_fingerprint_weighted)
