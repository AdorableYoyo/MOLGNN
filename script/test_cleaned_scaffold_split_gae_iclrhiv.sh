
#finetune_epochs=100
#split_ratio=0.9
# not train gae
#for dataset in JAK2 JAK1 JAK3
#do
#for split_ratio in 0.1 0.5 0.9
#do
#python jak_classification_random_split_GIN_GAE.py  \
#--dataset ${dataset}  \
#--finetune_epochs ${finetune_epochs}    \
#--featureencoding onehot  \
#--split_ratio ${split_ratio} \
#--device ${device}  > ${dataset}_epochs${epochs}_randomsplit_NON_GAE_GIN_GIAE${split_ratio}_output.log
#done
#done


#TRAIN gae

#computer=dls
computer=dgx
#data_splitting_method='random_split'
data_splitting_method='scaffold_split'
experiment_repeat_id=1230_1
stage=validation
split_ratio=0.8
#dataset=JAK1
#gae_extra_datasets='NOEXTRA'
#gae_extra_datasets=ZINC1k
#gae_extra_datasets=ZINC10k
#if [ $computer == dgx ]
#then
datapath=/Users/wuyou/MOLGNN_MTL/Jak_/
deepchem_datapath=/Users/wuyou/MOLGNN_MTL/DeepChem/
tflog=/Users/wuyou/MOLGNN_MTL/tflogs_1230
device=1
#fi
#gae_train_method=unfreeze
#gae_train_method=freeze_n_epoch
#gae_train_method=gradual_unfreeze
gae_train_method=static_fusing
pretrain_epochs=1
freeze_epochs=1
finetune_epochs=2
#finetune_epochs=200
#BACE BBBP ClinTox HIV
#BACE BBBP ClinTox HIV
#MUV Sider

for dataset in BACE
#for dataset in BACE BBBP ClinToxBalanced HIVBalanced
#for dataset in BACE BBBP
#for dataset in ICLR2020_HIV
#for dataset in HIV
#for dataset in BACE
#for dataset in ClinTox Sider
#for dataset in JAK1
#for dataset in ClinToxBalanced HIVBalanced
do
#for unsupervised_training_branches in adjacency_matrix_fingerprint adjacency_matrix fingerprint
for unsupervised_training_branches in fingerprint
do
nohup time 
python MoLGNN.py  \
--dataset ${dataset}  \
--stage ${stage} \
--data_splitting_method ${data_splitting_method} \
--split_ratio ${split_ratio} \
--train_gae \
--experiment_repeat_id ${experiment_repeat_id} \
--unsupervised_training_branches ${unsupervised_training_branches} \
--pretrain_epochs  ${pretrain_epochs} \
--freeze_epochs ${freeze_epochs} \
--finetune_epochs  ${finetune_epochs} \
--gae_train_method  ${gae_train_method} \
--featureencoding onehot  \
--split_ratio ${split_ratio} \
--datapath ${datapath}  \
--deepchem_datapath ${deepchem_datapath} \
--tflog  ${tflog}  \
--device ${device}  > ./logs_1230/${dataset}FP_${experiment_repeat_id}_epochs${epochs}_${data_splitting_method}_${split_ratio}_${gae_train_method}_${unsupervised_training_branches}_output.log 2>&1 &
echo "done ${dataset} ${split_ratio}"
done
done
