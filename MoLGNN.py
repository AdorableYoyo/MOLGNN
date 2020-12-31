"""
jimmy shen 
June 10, 2020
This code is used to train the JAK model based on random train and test dataset splitting.
This code is modified from "jak_classification_random_split.py"

For this version, a network contain two outputs will be designed: autoencoder, classification. We apply a weight for each part
gae_weight, classification_weight 
if we set the gae_weight as 0, it means that the autoencoder branch is turned off
meanwhile, if we set the classification_weight as 0, it means that the classification part will be turned off.
By doing this, we can easily pretrain GIAE which is a graph autoencoder

Updated by jimmy shen July 2020. 
The mainly updates is adding the fingerprint into the whole pipeline, right now, we have three heads in the pipeline
1. predict the adjanct matrix by using the VGAE
2. predict the fingerprint by using the VGAE
3. classify the graph by using GIN
In order to make less change to the original codebase, the first brach is keeping on using the name of gae.
Based on this, we will have 3 braches accordingly by using the name of
branch 1: *gae*
branch 2: *fingerprint*
branch 3: *classification*
such as when we control the final loss of the whole network, we are using weights for each branche, the 
variable names of the weight related to those 3 branches are:
 'gae_weight_rt, fingerprint_weight_rt, classification_weight_rt'. Here rt means real time. 


 Updated by Yoyo Dec 2020.
 deleted extradataset
 created train.py eval.py , freeze.py 
 deleted gae_train_method: unfreeze , freeze_n_epoch

"""

import sys, os
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from jakdt import GINDataset, DeepChemDataset, _MultipleLabelDatasets
from deepchem_dataloader import DeepChemDatasetPG
from jak_dataloader import GraphDataLoader, GraphDataLoaderSplit, collate
from parser_MoLGNN import Parser
from vgin import GIN, GIN_VGAE
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from optimizer import loss_function as variational_loss_function
from torch.nn.functional import binary_cross_entropy_with_logits as BCELoss
from datetime import datetime
from train import train
from eval import eval_net 
from freeze import freeze_model_weights,unfreeze_model_weights
# most deepChem datasets have the pytorch geometric format graph data
_DeepChemDatasets = {'BACE',
                     'BACEFP',
                     'BBBP',
                     'BBBPFP',
                     'ClinTox', # this dataset only has one label
                     'ClinToxBalanced',
                     'ClinTox_twoLabel', # this dataset has two labels
                     'ClinToxFP',
                     'HIV',
                     'HIVBalanced',
                     'HIVFP',
                     'MUV',
                     'MUVFP',
                     'Sider',
                     'SiderFP'}
for dataset in list(_DeepChemDatasets):
    _DeepChemDatasets.add("ICLR2020_" + dataset)

def main(args):
    pretrain_epochs = args.pretrain_epochs
    finetune_epochs = args.finetune_epochs
    freeze_epochs = args.freeze_epochs
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    if args.train_gae:
        if args.gae_train_method == 'static_fusing':
            epochs = finetune_epochs 
        else:
            epochs = pretrain_epochs + finetune_epochs
        experiment_id = '_'.join(['gae',
                                  args.unsupervised_training_branches,
                                  args.stage,
                                  args.featureencoding,
                                  str(epochs), 
                                  str(args.final_dropout),
                                  "split"+str(args.split_ratio),
                                  "hidden"+str(args.hidden_dim),
                                  args.gae_train_method,
                                  str(pretrain_epochs),
                                  str(freeze_epochs),
                                  str(finetune_epochs),
                                  str(args.gae_weight),
                                  str(args.classification_weight)])
    else:
        epochs = finetune_epochs
        experiment_id = '_'.join(['NON_gae',args.stage, args.featureencoding, str(epochs), str(args.final_dropout), "split"+str(args.split_ratio)])
    tflogs =args.tflog +'/'+args.data_splitting_method+args.experiment_repeat_id+'/'+str(args.split_ratio)+'/'+args.dataset+'_gin/'+experiment_id
    if not os.path.exists(tflogs):
        os.makedirs(tflogs)
    args.filename = tflogs+"res.txt"
    writer = SummaryWriter(log_dir=tflogs)
    # set up seeds, args.seed supported
    torch.manual_seed(seed=args.seed)
    np.random.seed(seed=args.seed)

    is_cuda = not args.disable_cuda and torch.cuda.is_available()

    if is_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)
        device = torch.device("cuda:" + str(args.device))
        #torch.cuda.set_device(args.device)
        args.device = torch.device("cuda:" + str(args.device))
        torch.cuda.manual_seed_all(seed=args.seed)
    else:
        args.device = torch.device("cpu")
    extra_datasets, gae_extra_dataset_loaders = [], []

   
    scaffold_split_idx_path = args.datapath+args.dataset+'/'
    if args.dataset in _DeepChemDatasets:
        # Full datasets is used to generate one hoe encoding.
        # the reason of this full_datasets does not contain the args.dataset is
        # in the _get_ONEHOTENCODING_CODEBOOKS method of DeepChemDataset
        # during the Onehotencoding codebooks generation process, the args.dataset's data and
        # extra dataset's data are combined together. So here is not needed.
        full_datasets = extra_datasets
        print("full dataset is ", full_datasets)
        # for the Pytorch Geometric Format Datasets, we have two kinds of datasets
        # 1. datasets with graph labels such as BACE, BBBp
        # 2. datasets without graph labels such as ZINC1k.
        # for the first kind, if the fingerpinter is going to be used, we need to use two inputs:
        # one input is from "deepchem_datapath" where the label is the graph label
        # the other input is from "deepchem_fingerprint_datapath" where the label is the finger printer.
        # For the second kind, since we don't have the graph label, we only have one dataset which contains the finger printer as label.
        
        # this dataset contains the graph label as the label
        deepchem_datapath = args.deepchem_datapath + args.dataset + '/'
        scaffold_split_idx_path = deepchem_datapath
        # this dataset contains the fingerpinter as the label
        deepchem_fingerprint_datapath = args.deepchem_datapath + args.dataset+'FP'+ '/'
        assert os.path.exists(deepchem_fingerprint_datapath), "currently version code requires the exists of the fingerprint data"
        dataset_pg = DeepChemDatasetPG(deepchem_datapath)
        dataset_fingerprint_pg = DeepChemDatasetPG(deepchem_fingerprint_datapath)
        train_val_test_dataset = DeepChemDataset(
                               dataset_pg,
                               dataset_fingerprint_pg,
                               datapath=args.datapath,
                               self_loop=not args.learn_eps, 
                               featureencoding=args.featureencoding,
                               full_datasets=full_datasets,
                               main_dataset=args.dataset) # the reason that we need the full datasets information as we need align features from all datasets to do the one hot encoding
      
    else:
        full_datasets = [args.dataset]+extra_datasets
        print("full dataset is ", full_datasets)
        train_val_test_dataset = GINDataset(
                               name=args.dataset,
                               self_loop=not args.learn_eps,
                               datapath=args.datapath,
                               full_datasets=full_datasets,
                               dataset=args.dataset,
                               featureencoding=args.featureencoding)
    # validloader is None if we don't have this split.
    trainloader, validloader, testloader = GraphDataLoaderSplit(
            dataset=train_val_test_dataset,
            stage=args.stage,
            batch_size=args.batch_size, 
            device=args.device,
            collate_fn=collate,
            split_ratio=args.split_ratio,
            shuffle=True,
            seed=args.seed,
            data_splitting_method=args.data_splitting_method,
            scaffold_split_idx_path=scaffold_split_idx_path,
            main_dataset=args.dataset,USING_CORRECT_scaffold_split_implementation=False).get_data_loader()

        
    model = GIN_VGAE(
        args.num_layers,
        args.num_mlp_layers,
        train_val_test_dataset.dim_nfeats,
        args.hidden_dim,
        train_val_test_dataset.fingerprint_dim,
        train_val_test_dataset.gclasses,
        args.final_dropout,
        args.learn_eps,
        args.graph_pooling_type,
        args.neighbor_pooling_type).to(args.device)
    model_dict = model.state_dict()
    print('-'*20 + 'model'+'-'*20)
    for key in sorted(model_dict.keys()):
        parameter = model_dict[key]
        print(key)
        print(parameter.size())
    print('-'*40)
    criterion_gae = BCELoss
    if args.dataset in _MultipleLabelDatasets:
        print(args.dataset, "BCELoss is used for multiple label prediction")
        criterion_classification = BCELoss # defaul reduce is true
    else:
        criterion_classification = nn.CrossEntropyLoss()  # defaul reduce is true
    criterion_fingerprint = BCELoss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # it's not cost-effective to hanle the cursor and init 0
    # https://stackoverflow.com/a/23121189
    ############ training ############
    tbar = tqdm(range(epochs), unit="epoch", position=3, ncols=0, file=sys.stdout)
    vbar = tqdm(range(epochs), unit="epoch", position=4, ncols=0, file=sys.stdout)
    lrbar = tqdm(range(epochs), unit="epoch", position=5, ncols=0, file=sys.stdout)

    for epoch, _, _ in zip(tbar, vbar, lrbar):
   

                # for the pretrain stage, the valid and test datasplit should also be trained
                # However, only the unsupervised branches are trained (the VGAE branches)
                # the supervised branch(graph classification) will not be trained to avoid the data leaking.
                # We disable the graph classification training by setting classification_weight_rt as 0
                # the trainloader will always be trained in both pretrain stage and fintune stage.
                
        if args.train_gae and args.gae_train_method == 'gradual_unfreeze':
            print('gae based on freeze_n_epoch will be applied')
            # pretrain stage, validation also needs to be trained
            stage_key = {0:"01234",
                         1:"0123",
                         2:"012",
                         3:"01",
                         4:"0"}
            learning_rates_stages = {0:args.lr,
                                    1:args.lr,
                                    2:args.lr/2.,
                                    3:args.lr/2.,
                                    4:args.lr/4.}
            epochs_finetune =  epoch-args.pretrain_epochs
            
            if epochs_finetune>=0 and (epochs_finetune%25==0):
                frozen_stage = epochs_finetune//25
                if 0<=frozen_stage<=4:
                    # unfrozen first
                    model = unfreeze_model_weights(model)
                    # reinitialize the scheduler
                    model = freeze_model_weights(model, layers=stage_key[frozen_stage])
                    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rates_stages[frozen_stage])
                    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
                elif frozen_stage==5:
                    model = unfreeze_model_weights(model)
                    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr/4.)
                    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
            
              
            if epoch < args.pretrain_epochs:
                if args.unsupervised_training_branches == "adjacency_matrix":
                    gae_weight_rt, fingerprint_weight_rt = 1.0, 0.0
                elif args.unsupervised_training_branches == "fingerprint":
                    gae_weight_rt, fingerprint_weight_rt = 0.0, 1.0
                elif args.unsupervised_training_branches=="adjacency_matrix_fingerprint":
                    gae_weight_rt, fingerprint_weight_rt = 0.5, 0.5
                else:
                    raise
                #gae_weight_rt = 1.0
                classification_weight_rt = 0.0
                train(args,
                      model, 
                      validloader, 
                      optimizer, 
                      criterion_gae, 
                      criterion_classification, 
                      criterion_fingerprint,
                      epoch,
                      gae_weight_rt=gae_weight_rt,
                      classification_weight_rt=0.0,
                      fingerprint_weight_rt=fingerprint_weight_rt)
                train(args,
                      model, 
                      testloader, 
                      optimizer, 
                      criterion_gae, 
                      criterion_classification, 
                      criterion_fingerprint,
                      epoch,
                      gae_weight_rt=gae_weight_rt,
                      classification_weight_rt=0.0,
                      fingerprint_weight_rt=fingerprint_weight_rt)
                
            # fine tune stage, set gae_weight to 0 to turn it off
            else:
                gae_weight_rt = 0.0
                classification_weight_rt = 1.0
                fingerprint_weight_rt= 0.0
           
        elif args.train_gae and args.gae_train_method == 'static_fusing':
            gae_weight_rt = args.gae_weight
            classification_weight_rt = args.classification_weight
            fingerprint_weight_rt= args.fingerprint_weight
            train(args,
                      model, 
                      validloader, 
                      optimizer, 
                      criterion_gae, 
                      criterion_classification, 
                      criterion_fingerprint,
                      epoch,
                      gae_weight_rt=gae_weight_rt,
                      classification_weight_rt=0.0,
                      fingerprint_weight_rt=fingerprint_weight_rt)
            train(args,
                      model, 
                      testloader, 
                      optimizer, 
                      criterion_gae, 
                      criterion_classification, 
                      criterion_fingerprint,
                      epoch,
                      gae_weight_rt=gae_weight_rt,
                      classification_weight_rt=0.0,
                      fingerprint_weight_rt=fingerprint_weight_rt)
            
        else:
            gae_weight_rt = 0.0
            classification_weight_rt = 1.0
            fingerprint_weight_rt= 0.0
        train(args, 
              model, 
              trainloader, 
              optimizer, 
              criterion_gae,
              criterion_classification, 
              criterion_fingerprint,
              epoch,gae_weight_rt=gae_weight_rt,
              classification_weight_rt=classification_weight_rt,
              fingerprint_weight_rt=fingerprint_weight_rt)    
                   
        scheduler.step()

        

        ret = eval_net(args, model, trainloader, criterion_gae, criterion_classification,criterion_fingerprint,gae_weight_rt=gae_weight_rt,
                  classification_weight_rt=classification_weight_rt, fingerprint_weight_rt=fingerprint_weight_rt)
        (train_loss,
         train_loss_gae_original,
         train_loss_classification_original,
         train_loss_fingerprint_original,
         train_loss_gae_weighted,
         train_loss_classification_weighted,
         train_loss_fingerprint_weighted,
         train_acc,
         train_roc_score,
         train_ap_score,
         train_roc_score_micro,
         train_ap_score_micro) = ret
        tbar.set_description(
            'train set - average loss: {:.4f}, accuracy: {:.0f}%'
            .format(train_loss, 100. * train_acc))
        # some split we may only have train and test
        if validloader is not None:
            print('validation dataset evaluation ...')
            ret = eval_net(args,
                           model,
                           validloader,
                           criterion_gae,
                           criterion_classification,
                           criterion_fingerprint,
                           gae_weight_rt=gae_weight_rt,
                           classification_weight_rt=classification_weight_rt,
                           fingerprint_weight_rt=fingerprint_weight_rt)
            (valid_loss,
             valid_loss_gae_original,
             
             _loss_classification_original,
             valid_loss_fingerprint_original,
             valid_loss_gae_weighted,
             valid_loss_classification_weighted,
             valid_loss_fingerprint_weighted,
             valid_acc,
             valid_roc_score,
             valid_ap_score,
             valid_roc_score_micro,
             valid_ap_score_micro) = ret
        print('test dataset evaluation ...')
        ret = eval_net(args, model, testloader, criterion_gae, criterion_classification,criterion_fingerprint,gae_weight_rt=gae_weight_rt,
                  classification_weight_rt=classification_weight_rt, fingerprint_weight_rt=fingerprint_weight_rt)
        (test_loss,
         test_loss_gae_original,
         test_loss_classification_original,
         test_loss_fingerprint_original,
         test_loss_gae_weighted,
         test_loss_classification_weighted,
         test_loss_fingerprint_weighted,
         test_acc,
         test_roc_score,
         test_ap_score,
         test_roc_score_micro,
         test_ap_score_micro) = ret 
        print("#############################################################TEST LOSS GAE WEIGHTED##################",test_loss_gae_weighted)
        vbar.set_description(
            'valid set - average loss: {:.4f}, accuracy: {:.0f}%'
            .format(test_loss, 100. * test_acc))
        writer.add_scalar('Loss/train_WHOLE', train_loss , epoch)
        writer.add_scalar('Loss/train_loss_gae_original_WHOLE', train_loss_gae_original , epoch)
        writer.add_scalar('Loss/train_loss_classification_original_WHOLE', train_loss_classification_original , epoch)
        writer.add_scalar('Loss/train_loss_fingerprint_original_WHOLE', train_loss_fingerprint_original , epoch)
        writer.add_scalar('Loss/train_loss_gae_weighted_WHOLE', train_loss_gae_weighted , epoch)
        writer.add_scalar('Loss/train_loss_classification_weighted_WHOLE', train_loss_classification_weighted , epoch)
        writer.add_scalar('Loss/train_loss_fingerprint_weighted_WHOLE', train_loss_fingerprint_weighted , epoch)
        writer.add_scalar('Accuracy/train_WHOLE', train_acc , epoch)
        writer.add_scalar('roc_score/train_WHOLE', train_roc_score , epoch)
        writer.add_scalar('ap_score/train_WHOLE', train_ap_score , epoch)
        writer.add_scalar('roc_score_micro/train_WHOLE', train_roc_score_micro , epoch)
        writer.add_scalar('ap_score_micro/train_WHOLE', train_ap_score_micro , epoch)
        # valid
        if validloader is not None: 
            writer.add_scalar('Loss/valid_WHOLE', valid_loss , epoch)
            writer.add_scalar('Loss/valid_loss_gae_original_WHOLE', valid_loss_gae_original , epoch)
            writer.add_scalar('Loss/valid_loss_classification_original_WHOLE', valid_loss_classification_original , epoch)
            writer.add_scalar('Loss/valid_loss_fingerprint_original_WHOLE', valid_loss_fingerprint_original , epoch)
            writer.add_scalar('Loss/valid_loss_gae_weighted_WHOLE', valid_loss_gae_weighted , epoch)
            writer.add_scalar('Loss/valid_loss_classification_weighted_WHOLE', valid_loss_classification_weighted , epoch)
            writer.add_scalar('Loss/valid_loss_fingerprint_weighted_WHOLE', valid_loss_fingerprint_weighted , epoch)
            writer.add_scalar('Accuracy/valid_WHOLE', valid_acc , epoch)
            writer.add_scalar('roc_score/valid_WHOLE', valid_roc_score , epoch)
            writer.add_scalar('ap_score/valid_WHOLE', valid_ap_score , epoch)
            writer.add_scalar('roc_score_micro/valid_WHOLE', valid_roc_score_micro , epoch)
            writer.add_scalar('ap_score_micro/valid_WHOLE', valid_ap_score_micro , epoch)
        # test
        writer.add_scalar('Loss/test_WHOLE', test_loss , epoch)
        writer.add_scalar('Loss/test_loss_gae_original_WHOLE', test_loss_gae_original , epoch)
        writer.add_scalar('Loss/test_loss_classification_original_WHOLE', test_loss_classification_original , epoch)
        writer.add_scalar('Loss/test_loss_fingerprint_original_WHOLE', test_loss_fingerprint_original , epoch)
        writer.add_scalar('Loss/test_loss_gae_weighted_WHOLE', test_loss_gae_weighted , epoch)
        writer.add_scalar('Loss/test_loss_classification_weighted_WHOLE', test_loss_classification_weighted , epoch)
        writer.add_scalar('Loss/test_loss_fingerprint_weighted_WHOLE', test_loss_fingerprint_weighted , epoch)
        writer.add_scalar('Accuracy/test_WHOLE', test_acc , epoch)
        writer.add_scalar('roc_score/test_WHOLE', test_roc_score , epoch)
        writer.add_scalar('ap_score/test_WHOLE', test_ap_score , epoch)
        writer.add_scalar('roc_score_micro/test_WHOLE', test_roc_score_micro, epoch)
        writer.add_scalar('ap_score_micro/test_WHOLE', test_ap_score_micro, epoch)
        if (not args.train_gae) or (args.train_gae and epoch>=args.pretrain_epochs) or (args.train_gae and  args.gae_train_method == 'static_fusing'):
            if not args.train_gae or (args.train_gae and  args.gae_train_method == 'static_fusing'):
                epoch_ = epoch
            elif args.train_gae and epoch>=args.pretrain_epochs:
                epoch_ = epoch - args.pretrain_epochs
            writer.add_scalar('Loss/train', train_loss , epoch_)
            writer.add_scalar('Loss/train_loss_gae_original', train_loss_gae_original , epoch_)
            writer.add_scalar('Loss/train_loss_classification_original', train_loss_classification_original , epoch_)
            writer.add_scalar('Loss/train_loss_fingerprint_original', train_loss_fingerprint_original , epoch_)
            writer.add_scalar('Loss/train_loss_gae_weighted', train_loss_gae_weighted , epoch_)
            writer.add_scalar('Loss/train_loss_classification_weighted', train_loss_classification_weighted , epoch_)
            writer.add_scalar('Loss/train_loss_fingerprint_weighted', train_loss_fingerprint_weighted , epoch_)
            writer.add_scalar('Accuracy/train', train_acc , epoch_)
            writer.add_scalar('roc_score/train', train_roc_score , epoch_)
            writer.add_scalar('ap_score/train', train_ap_score , epoch_)
            writer.add_scalar('roc_score_micro/train', train_roc_score_micro , epoch_)
            writer.add_scalar('ap_score_micro/train', train_ap_score_micro , epoch_)
            # valid
            if validloader is not None: 
                writer.add_scalar('Loss/valid', valid_loss , epoch_)
                writer.add_scalar('Loss/valid_loss_gae_original', valid_loss_gae_original , epoch_)
                writer.add_scalar('Loss/valid_loss_classification_original', valid_loss_classification_original , epoch_)
                writer.add_scalar('Loss/valid_loss_fingerprint_original', valid_loss_fingerprint_original , epoch_)
                writer.add_scalar('Loss/valid_loss_gae_weighted', valid_loss_gae_weighted , epoch_)
                writer.add_scalar('Loss/valid_loss_classification_weighted', valid_loss_classification_weighted , epoch_)
                writer.add_scalar('Loss/valid_loss_fingerprint_weighted', valid_loss_fingerprint_weighted , epoch_)
                writer.add_scalar('Accuracy/valid', valid_acc , epoch_)
                writer.add_scalar('roc_score/valid', valid_roc_score , epoch_)
                writer.add_scalar('ap_score/valid', valid_ap_score , epoch_)
                writer.add_scalar('roc_score_micro/valid', valid_roc_score_micro , epoch_)
                writer.add_scalar('ap_score_micro/valid', valid_ap_score_micro , epoch_)
            # test
            writer.add_scalar('Loss/test', test_loss , epoch_)
            writer.add_scalar('Loss/test_loss_gae_original', test_loss_gae_original , epoch_)
            writer.add_scalar('Loss/test_loss_classification_original', test_loss_classification_original , epoch_)
            writer.add_scalar('Loss/test_loss_fingerprint_original', test_loss_fingerprint_original , epoch_)
            writer.add_scalar('Loss/test_loss_gae_weighted', test_loss_gae_weighted , epoch_)
            writer.add_scalar('Loss/test_loss_classification_weighted', test_loss_classification_weighted , epoch_)
            writer.add_scalar('Loss/test_loss_fingerprint_weighted', test_loss_fingerprint_weighted , epoch_)
            writer.add_scalar('Accuracy/test', test_acc , epoch_)
            writer.add_scalar('roc_score/test', test_roc_score , epoch_)
            writer.add_scalar('ap_score/test', test_ap_score , epoch_)
            writer.add_scalar('roc_score_micro/test', test_roc_score_micro , epoch_)
            writer.add_scalar('ap_score_micro/test', test_ap_score_micro , epoch_)
        if args.filename:
            with open(args.filename, 'a') as f:
                f.write('%s %s %s %s' % (
                    args.dataset,
                    args.learn_eps,
                    args.neighbor_pooling_type,
                    args.graph_pooling_type
                ))
                f.write("\n")
                if validloader is None:
                    f.write("%f %f %f %f" % (
                        train_loss,
                        train_acc,
                        test_loss,
                        test_acc
                    ))
                    f.write("\n")
                    f.write("%f %f %f %f" % (
                        train_roc_score,
                        train_ap_score,
                        test_roc_score,
                        test_ap_score
                    ))
                    f.write("\n")
                else:
                    f.write("%f %f%f %f %f %f" % (
                        train_loss,
                        train_acc,
                        valid_loss,
                        valid_acc,
                        test_loss,
                        test_acc
                    ))
                    f.write("\n")
                    f.write("%f %f %f %f %f %f" % (
                        train_roc_score,
                        train_ap_score,
                        valid_roc_score,
                        valid_ap_score,
                        test_roc_score,
                        test_ap_score
                    ))
                    f.write("\n")

        lrbar.set_description(
            "Learning eps with learn_eps={}: {}".format(
                args.learn_eps, [layer.eps.data.item() for layer in model.ginlayers]))

    tbar.close()
    vbar.close()
    lrbar.close()


if __name__ == '__main__':
    
    parser = Parser(description='VGAE_GIAE').parser
    #parser.add_argument('--tflog', '-t', type=str, default='/raid/home/jimmyshen/repos/jak_classification_logs/', help='result directry')
    #parser.add_argument('--featureencoding', '-f', type=str, default='onehot', help='using onhot encoding or using original feature')
    #parser.add_argument('--split_ratio', type=float, default=0.9, help='train test split ratio (default: 0.9)')
    args = parser.parse_args()
    print('show all arguments configuration...')
    print(args)
    

    main(args)
