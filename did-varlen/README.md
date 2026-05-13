# did-varlen

<!-- 
## Pretrained Models

|Model|Loss function|Total model size|
|:---|:---:|:---:|
|[did-varlen-small-stories]( )| DISE | 132M | -->

## Training

```bash
# configure single machine with multiple GPUs
torchrun --nnodes 1 run_train_ddp.py

# or, configure multiple machines with multiple GPUs 
torchrun --nnodes NNODES --nproc_per_node NPROC_PER_NODE --master-addr MASTER_ADDR --node-rank NODE_RANK run_train_ddp.py 

# DID training
torchrun --nproc-per-node=8 train_ddp.py model=small_did training.loss_type=DISE final_layer=csoftmax model.length=1024 training.n_iters=60001 data.train=tiny_roc_stories data.valid=tiny_roc_stories training.accum=1

# RADD training
torchrun --nproc-per-node=8 train_ddp.py model=small_radd training.loss_type=t_DCE  model.length=1024 training.n_iters=60001 data.train=tiny_roc_stories data.valid=tiny_roc_stories training.snapshot_freq=10000 training.accum=2 
```


## Sampling


```bash
## unconditional generation
python run_sample.py --model_path $did1024 --steps 64 --total_num 64 --batch_size 32
python run_sample_radd.py --model_path $ra1024 --length 1024 --steps 64 --total_num 64 --batch_size 32
python run_sample_radd.py --model_path $ra512 --length 512 --steps 64 --total_num 64 --batch_size 32

```

## Acknowledgements

This repository is built upon [RADD](https://github.com/ML-GSAI/RADD).

