The original repo link. https://github.com/liketheflower/graph_classification_jak

# Environment setup

we are using conda to manage the package

## Install a separate environment by conda

```
conda create --name gnn python=3.7
```

## get pytorch geometric

activate the installed environment by

```
 source activate gnn
```

then install the pytorch geometric, during this process, torch will also be installed

```
install_torch_geo.sh
```

## Install DGL

```
conda install -c dglteam dgl=0.4.3post2
```

## Install tensorflow

```
conda install -c conda-forge tensorflow==2.1.0
```

# Molgnn static fusion based on scaffold split BACE data set

## How to run

```
bash script/test_cleaned_scaffold_split_gae_iclrhiv.sh
```
