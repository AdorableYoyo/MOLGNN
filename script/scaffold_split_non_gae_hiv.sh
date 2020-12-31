# This shell script is used to train and test the NON GAE branch

#computer=dls
computer=dgx
#data_splitting_method='random_split'
data_splitting_method='scaffold_split'
stage=validation
split_ratio=0.8
#dataset=JAK1
gae_extra_datasets='NOEXTRA'
#gae_extra_datasets=ZINC1k
#gae_extra_datasets=ZINC10k
if [ $computer == dgx ]
then
   datapath=/raid/home/jimmyshen/JAK_/original/
   deepchem_datapath=/raid/home/jimmyshen/DeepChem/
   tflog=/raid/home/jimmyshen/repos/jak_classification_logs_gin_gae/gae_fingerprinter/
   device=0
fi
#gae_train_method=unfreeze
#gae_train_method=freeze_n_epoch
gae_train_method=gradual_unfreeze
#gae_train_method=static_fusing
pretrain_epochs=100
freeze_epochs=50
finetune_epochs=100
#finetune_epochs=200
#BACE BBBP ClinTox HIV
#MUV Sider

#for dataset in BACE:wq



#for dataset in BACE BBBP ClinToxBalanced HIVBalanced
#for experiment_repeat_id in 0 1 2 3 4
for experiment_repeat_id in 11 12 13 14
do
#for dataset in BACE BBBP
for dataset in HIV
#for dataset in BBBP
#for dataset in JAK1 JAK2 JAK3
#for dataset in ClinToxBalanced HIVBalanced
do
nohup time 
python MoLGNN.py  \
--dataset ${dataset}  \
--stage ${stage} \
--data_splitting_method ${data_splitting_method} \
--split_ratio ${split_ratio} \
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
--device ${device}  > ./logs/${dataset}_${experiment_repeat_id}_FP_epochs${epochs}_${data_splitting_method}_NON_GAE.log 2>&1 &
echo "done ${dataset} ${split_ratio}"
done
done
