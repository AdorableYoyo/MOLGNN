import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from parser_MoLGNN import Parser
import torch
from jakdt import  _MultipleLabelDatasets


def eval_net(args, 
             net, 
             dataloader, 
             criterion_gae, 
             criterion_classification,
             criterion_fingerprint,
             gae_weight_rt=0,
             classification_weight_rt=1.0,
             fingerprint_weight_rt=1.0):
    net.eval()

    total = 0
    #total_loss = 0
    running_loss_gae_original = 0
    running_loss_classification_original = 0
    running_loss_fingerprint_original = 0
    
    running_loss_gae_weighted = 0
    running_loss_classification_weighted = 0
    running_loss_fingerprint_weighted = 0
    
    running_loss = 0
    
    total_correct = 0
    y_true, y_pred = [], []
    for data in dataloader:
        graphs, labels, fingerprints= data
        fingerprints = fingerprints.to(args.device)
        fingerprint_gt = fingerprints
        if  args.dataset in _MultipleLabelDatasets:
            labels = labels.double()
        this_labels = labels.numpy().tolist()
        y_true += this_labels
        feat = graphs.ndata['attr'].to(args.device)
        #fingerprint_gt = graphs.ndata['fingerprint'].to(args.device)
        labels = labels.to(args.device)
        total += len(labels)
        
        adj = graphs.adjacency_matrix().to_dense()
        adj_np = adj.numpy()
        adj = sp.csr_matrix(adj_np)
        adj_label = torch.FloatTensor(adj.toarray()).to(args.device)
        
        pos_weight = torch.Tensor([float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()]).to(args.device)
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        adj_rec, _, _, score_over_layer_classification, fingerprint_rec = net(graphs, feat)
        #norm*loss_function(adj_logits, adj_label, pos_weight=pos_weight)
        loss_gae = norm*criterion_gae(adj_rec, adj_label, pos_weight=pos_weight)
        loss_classification = criterion_classification(score_over_layer_classification, labels)
        loss_fingerprint = criterion_fingerprint(fingerprint_rec, fingerprint_gt)
   
        running_loss_gae_original += loss_gae.item()  * len(labels)
        running_loss_classification_original += loss_classification.item()  * len(labels)
        running_loss_fingerprint_original += loss_fingerprint.item() * len(labels)
        
        # rt is short for real time
        loss_gae_weighted = gae_weight_rt*loss_gae
        loss_classification_weighted = classification_weight_rt*loss_classification
        loss_fingerprint_weighted = fingerprint_weight_rt*loss_fingerprint
      
        
        running_loss_gae_weighted += loss_gae_weighted.item() 
        running_loss_classification_weighted += loss_classification_weighted.item()
        running_loss_fingerprint_weighted += loss_fingerprint_weighted.item()
    
        
        
        loss = loss_gae_weighted+loss_classification_weighted +loss_fingerprint_weighted
        running_loss += loss.item() 
       
        
        assert score_over_layer_classification.ndim==2
        if  args.dataset in _MultipleLabelDatasets:
            this_y_pred_after_sigmoid = torch.nn.Sigmoid()(score_over_layer_classification)
            this_y_pred_after_sigmoid = this_y_pred_after_sigmoid.detach().cpu().numpy()
            #this_y_pred_after_sigmoid = this_y_pred_after_sigmoid[:,1]
            y_pred += this_y_pred_after_sigmoid.tolist()
        else:
            this_y_pred_after_softmax = torch.nn.Softmax(dim=1)(score_over_layer_classification)
            this_y_pred_after_softmax = this_y_pred_after_softmax.detach().cpu().numpy()
            #this_y_pred_after_softmax = this_y_pred_after_softmax[:,0]
            y_pred += this_y_pred_after_softmax.tolist()
            _, predicted = torch.max(score_over_layer_classification.data, 1)
            # multiple labeling do not caculating the accuracy
            total_correct += (predicted == labels.data).sum().item()
        #loss = criterion(score_over_layer_classification, labels)
        # crossentropy(reduce=True) for default
        #running_loss += loss.item() * len(labels)
    if  args.dataset not in _MultipleLabelDatasets:
        one_hot_y_true = []
        #print("original y_true", y_true[:10])
        for true_label in y_true:
            if int(true_label) == 0:
                one_hot_y_true.append([1.0, 0.0])
            else:
                one_hot_y_true.append([0.0, 1.0])
        # print("one_hot_y_true first 10", one_hot_y_true[:10])
        y_true = one_hot_y_true
    labels_all = np.array(y_true)
    preds_all = np.array(y_pred)
    assert labels_all.shape == preds_all.shape
#    print("labels all shape", labels_all.shape)
 #   print("preds all", preds_all.shape)
  #  print("labels all first 10", labels_all[:10])
   # print("preds all first 10", preds_all[:10])
    roc_score = roc_auc_score(labels_all, preds_all)
    roc_score_micro = roc_auc_score(labels_all, preds_all, average='micro')
    ap_score = average_precision_score(labels_all, preds_all)
    ap_score_micro = average_precision_score(labels_all, preds_all, average='micro')
    #print('macro ap', ap_score)
    #print('micro ap', ap_score_micro)
    running_loss, acc = 1.0*running_loss / total, 1.0*total_correct / total
    # new added items
    running_loss_gae_original /= total
    running_loss_classification_original /= total
    running_loss_gae_weighted /= total
#    print('TEST RUNNING LOSS GAE WEIGHTED LAST',running_loss_gae_weighted)
    running_loss_classification_weighted /= total
    running_loss_fingerprint_weighted /= total


    net.train()

    return (running_loss, 
            running_loss_gae_original,
            running_loss_classification_original,
            running_loss_fingerprint_original,
            running_loss_gae_weighted,
            running_loss_classification_weighted,
            running_loss_fingerprint_weighted,
            acc, 
            roc_score, 
            ap_score,
            roc_score_micro,
            ap_score_micro)
