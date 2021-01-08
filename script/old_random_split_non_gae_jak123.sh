# This shell script is used to train and test the NON GAE branch

#computer=dls
computer=dgx
stage=test
data_splitting_method='random_split'
#data_splitting_method='scaffold_split'
experiment_repeat_id=101
#dataset=JAK1
gae_extra_datasets='NOEXTRA'
#gae_extra_datasets=ZINC1k
#gae_extra_datasets=ZINC10k
if [ $computer == dgx ]
then
   datapath=/Users/wuyou/VSC/graph_classification_jak/MoLGNN/Jak_/
   deepchem_datapath=/raid/home/jimmyshen/DeepChem/
   tflog=/Users/wuyou/VSC/graph_classification_jak/MoLGNN/logs
   device=6
fi
#gae_train_method=unfreeze
#gae_train_method=freeze_n_epoch
gae_train_method=gradual_unfreeze
#gae_train_method=static_fusing
pretrain_epochs=1
freeze_epochs=5
finetune_epochs=1
#finetune_epochs=200
#BACE BBBP ClinTox HIV
#MUV Sider

#for dataset in BACE
#for dataset in BACE BBBP ClinToxBalanced HIVBalanced
#for dataset in BACE
#for dataset in BBBP
for dataset in JAK1 
#JAK2 JAK3
#for dataset in ClinToxBalanced HIVBalanced
do
for split_ratio in 0.1 
#0.5 0.9
#for split_ratio in 0.1
do
nohup time 
python MoLGNN.py  \
--dataset ${dataset}  \
--stage ${stage} \
--data_splitting_method ${data_splitting_method} \
--experiment_repeat_id ${experiment_repeat_id} \
--pretrain_epochs  ${pretrain_epochs} \
--freeze_epochs ${freeze_epochs} \
--finetune_epochs  ${finetune_epochs} \
--gae_extra_datasets ${gae_extra_datasets} \
--gae_train_method  ${gae_train_method} \
--featureencoding onehot  \
--split_ratio ${split_ratio} \
--datapath ${datapath}  \
--deepchem_datapath ${deepchem_datapath} \
--tflog  ${tflog}  \
--device ${device}  > /Users/wuyou/VSC/graph_classification_jak/MoLGNN/logs/non_debug${dataset}FP_${experiment_repeat_id}_epochs${epochs}_${data_splitting_method}_${split_ratio}.log 2>&1 &
echo "done ${dataset} ${split_ratio}"
done
done
