"""Parser for arguments

Put all arguments in one file and group similar arguments
"""
import argparse


class Parser():

    def __init__(self, description):
        '''
           arguments parser
        '''
        self.parser = argparse.ArgumentParser(description=description)
        self.args = None
        self._parse()

    def _parse(self):
        # dataset
        self.parser.add_argument(
            '--dataset', type=str, default="JAK1",
            #choices=['JAK1', 'JAK2', 'JAK3', 'JAK123','BACE','BACEFP','BBBP','BBBPFP', 'ClinTox',  'ClinToxFP','ClinToxBalanced','HIV','HIVBalanced','Sider', 'MUV'],
            help='name of dataset (default: MUTAG)')
        self.parser.add_argument(
            '--batch_size', type=int, default=32,
            help='batch size for training and validation (default: 32)')
        self.parser.add_argument(
            '--fold_idx', type=int, default=0,
            help='the index(<10) of fold in 10-fold validation.')
        self.parser.add_argument(
            '--filename', type=str, default="",
            help='output file')
        self.parser.add_argument(
            '--stage', type=str, default="test",
            help='validation stage will include the validation steps. test stage will train the train+validation and test based on test dataset')

        # device
        self.parser.add_argument(
            '--disable_cuda', action='store_false',
            help='Disable CUDA')
        self.parser.add_argument(
            '--device', type=int, default=1,
            help='which gpu device to use (default: 0)')

        # net
        self.parser.add_argument(
            '--num_layers', type=int, default=5,
            help='number of layers (default: 5)')
        self.parser.add_argument(
            '--num_mlp_layers', type=int, default=2,
            help='number of MLP layers(default: 2). 1 means linear model.')
        self.parser.add_argument(
            '--hidden_dim', type=int, default=32,
            help='number of hidden units (default: 64)')

        # graph
        self.parser.add_argument(
            '--graph_pooling_type', type=str,
            default="mean", choices=["sum", "mean", "max"],
            help='type of graph pooling: sum, mean or max')
        self.parser.add_argument(
            '--neighbor_pooling_type', type=str,
            default="mean", choices=["sum", "mean", "max"],
            help='type of neighboring pooling: sum, mean or max')
        self.parser.add_argument(
            '--learn_eps', action="store_true",
            help='learn the epsilon weighting')

        # learning
        self.parser.add_argument(
            '--seed', type=int, default=0,
            help='random seed (default: 0)')
        self.parser.add_argument(
            '--pretrain_epochs', type=int, default=100,
            help='number of pretrain_epochs to train the GAE(default: 100)')
        self.parser.add_argument(
            '--finetune_epochs', type=int, default=100,
            help='number of epochs to fine tune the classification network(default: 100)')
        self.parser.add_argument(
            '--freeze_epochs', type=int, default=50,
            help='number of epochs to freeze during the fintuning process (default: 50) if we have pretrain_epochs:100, finetune_epochs:100 then the freeze will be from 100 to 150 ')
        self.parser.add_argument(
            '--lr', type=float, default=0.01,
            help='learning rate (default: 0.01)')
        self.parser.add_argument(
            '--final_dropout', type=float, default=0.0,
            help='final layer dropout (default: 0.5)')
        # new added agruments
        self.parser.add_argument(
            '--datapath', 
            type=str, 
            default='/raid/home/jimmyshen/JAK/original/', 
            help='datapath')
        
        
        self.parser.add_argument(
            '--deepchem_datapath', 
            type=str, 
            default='/raid/home/jimmyshen/DeepChem/', 
            help='deepchem_datapath')
        
        self.parser.add_argument(
            '--experiment_repeat_id', 
            type=str, 
            default='0', 
            help='experiment may be repeated multiple times such as 10 times, it 10 times, the experiemnet repeat id will be 0, 1, ...9')
            
        self.parser.add_argument(
            '--data_splitting_method', 
            type=str, 
            default='random_split', 
            help='currently support random_split or scaffold_split')
        self.parser.add_argument(
            '--tflog', '-t', 
            type=str, 
            default='/raid/home/jimmyshen/repos/jak_classification_logs_gin_gae/', 
            help='result directry')
        
        self.parser.add_argument(
            '--featureencoding', 
            '-f', 
            type=str, 
            default='onehot', 
            help='using onhot encoding or using original feature')
        self.parser.add_argument('--split_ratio', 
                                 type=float, 
                                 default=0.9, 
                                 help='train test split ratio (default: 0.9)')
        self.parser.add_argument(
            '--train_gae', action='store_true',
            help='whether train the unsupervised_training_branches, such as the gae(adjacency matrix) only, fingerprint only or both')
        
        #self.parser.add_argument('--gae_extra_datasets', 
                                 #type=str, 
                                 #default='NOEXTRA', # no extra dataset will be used by default
                                 #help='extra DATASETs used to train the GAE as the classification dataset including test will always be part ot the GAE training, if we have multiple dataset, it will be encoded as dataseta_datasetb ...')
        
        
        self.parser.add_argument(
            '--gae_train_method', 
                                 type=str, 
                                 default='unfreeze', 
                                 help='unfreeze, freeze_n_epoch, gradual_unfreeze, static_fusing, dynamic_fusing')
        self.parser.add_argument(
            '--unsupervised_training_branches', 
                                 type=str, 
                                 default='adjacency_matrix', 
                                 help='adjacency_matrix, fingerprint, adjacency_matrix_fingerprint')
        self.parser.add_argument('--gae_weight', 
                                 type=float, 
                                 default=0.25, 
                                 help='initial loss weight for the GAE part(default: 0.5). The used weight may be different to this one if the dynamic weight policy is applied')
        
        self.parser.add_argument('--classification_weight', 
                                 type=float, 
                                 default=0.5, 
                                 help='initial classification weight for the classification part(default: 0.5). The used weight may be different to this one if the dynamic weight policy is applied')
        
        self.parser.add_argument('--fingerprint_weight', 
                                 type=float, 
                                 default=0.25, 
                                 help='initial fingerprint weight for the fingerprint prediction part(default: 0.5). The used weight may be different to this one if the dynamic weight policy is applied')

        # done
        self.args = self.parser.parse_args()
