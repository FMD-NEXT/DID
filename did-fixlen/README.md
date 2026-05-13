# did-fixlen

<!-- 
## Pretrained Models

|Model|Loss function|Total model size|
|:---|:---:|:---:|
|[did-fixlen-small-owt]( )| DICE |162M|
|[did-fixlen-medium-owt]( )| DICE |405M| -->

## Training

```bash
# configure single machine with multiple GPUs
torchrun --nnodes 1 run_train_ddp.py

# or, configure multiple machines with multiple GPUs 
torchrun --nnodes NNODES --nproc_per_node NPROC_PER_NODE --master-addr MASTER_ADDR --node-rank NODE_RANK run_train_ddp.py 

# DID training
torchrun --nproc-per-node=8 train_ddp.py model=small_did ngpus=8 training.accum=1 training.loss_type=DICE final_layer=seqnorm training.n_iters=800001 
torchrun --nproc-per-node=8 train_ddp.py model=medium_did ngpus=8 training.accum=2 training.loss_type=DICE final_layer=seqnorm training.n_iters=800001
torchrun --nproc-per-node=8 train_ddp.py model=large_did ngpus=8 training.accum=2 training.loss_type=DICE final_layer=seqnorm training.n_iters=800001

# RADD training
torchrun --nproc-per-node=8 train_ddp.py model=small_radd ngpus=8 training.accum=2 training.loss_type=lambda_DCE training.n_iters=400001
torchrun --nproc-per-node=8 train_ddp.py model=medium_radd ngpus=8 training.accum=4 training.loss_type=lambda_DCE training.n_iters=400001
torchrun --nproc-per-node=8 train_ddp.py model=large_radd ngpus=8 training.accum=4 training.loss_type=lambda_DCE training.n_iters=400001
```

Note that, we found the 'simple' loss [1] is beneficial for the Gen. PPL metric, i.e. just removing the 1/t coefficient in the training objectives of DLMs, which could reduce the variance of training. 

With the 'simple' loss, the Gen. PPL could stably reach 114 for DID-small and 82 for DID-medium trained on the OpenWebText dataset, better than the reported results in our paper.

[1] Demystifying Diffusion Objectives: Reweighted Losses are Better Variational Bounds. Jiaxin Shi, Michalis K. Titsias.

## Evaluation

```bash 
bash eval_ppl_did.sh $model_path
bash eval_ppl_radd.sh $model_path
```

## Sampling


```bash
# unconditional generation
python run_sample_did.py --model_path $model_path --total_num 64 --batch_size 32 --steps 128 --schedule uniform
python run_sample_radd.py --model_path $model_path --total_num 64 --batch_size 32 --steps 128 --schedule uniform

# conditional generation
python run_sample_cond_did.py --model_path $model_path --total_num 64 --batch_size 32 --steps 128 --prompt_len 256
python run_sample_cond_radd.py --model_path $model_path --total_num 64 --batch_size 32 --steps 128 --prompt_len 256
```

## Acknowledgements

This repository is built upon [RADD](https://github.com/ML-GSAI/RADD).

