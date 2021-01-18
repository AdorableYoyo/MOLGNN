data_splitting_method='scaffold_split'
experiment_repeat_id=0115_1
stage=validation
split_ratio=0.8
#datapath=/raid/home/jimmyshen/JAK_/original/
#deepchem_datapath=/Users/wuyou/DeepChem/
#tflog=/Users/wuyou/tflogs

datapath=/Users/wuyou/MOLGNN_MTL/Jak_/
deepchem_datapath=/raid/home/yoyowu/new_DeepChem/
tflog=/raid/home/yoyowu/MOLGNN/logs

device=1

for gae_train_method in static_fusing dynamic_fusing 
#gae_train_method=static_fusing
do

pretrain_epochs=100
finetune_epochs=200


for dataset in ClinTox_twoLabel

do

for unsupervised_training_branches in adjacency_matrix_fingerprint
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
--finetune_epochs  ${finetune_epochs} \
--gae_train_method  ${gae_train_method} \
--featureencoding onehot  \
--split_ratio ${split_ratio} \
--datapath ${datapath}  \
--deepchem_datapath ${deepchem_datapath} \
--tflog  ${tflog}  \
--device ${device}  > ./outlogs0112/${dataset}FP_${experiment_repeat_id}_epochs${epochs}_${data_splitting_method}_${split_ratio}_${gae_train_method}_${unsupervised_training_branches}_output.log 2>&1 &
echo "done ${dataset} ${split_ratio}"
done
done
done
