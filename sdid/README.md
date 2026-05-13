# Scaling up DID-varlen Model to 1.1B Parameters

## Training

```shell
# pretrain on slimpajama
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=8 pretrain/train_did.py --model 1028 --flops 1600. 
    
# sft on gsm8k
torchrun --nproc_per_node=8 sft/finetune_did_gsm8k.py --model 1028 --pretrain_path $ckpt_path
```
Please download the augmented training [data](https://github.com/da03/implicit_chain_of_thought/blob/main/data/gsm8k/train.txt) and
put the `train.txt` file in `./data/gsm8k` for SFT.

## Evaluation

Commonsense reasoning by comparing the (bound of) conditional likelihood of different options.
```shell
# mdm
accelerate launch --multi_gpu --num_processes 8 evaluate_diff.py --tasks openbookqa,winogrande,piqa,social_iqa,arc_easy,arc_challenge,hellaswag,race --model did --batch_size 64 --model_args model_name=1028,ckpt_path=$ckpt_path,mc_num=1024,nll_type='mc',greddy=False

# did
accelerate launch --multi_gpu --num_processes 8 evaluate_did.py --tasks openbookqa,winogrande,piqa,social_iqa,arc_easy,arc_challenge,hellaswag,race --model did --batch_size 64 --model_args model_name=1028,ckpt_path=$ckpt_path,mc_num=1024,cond_nll=True,per_token_loss=False
```

GSM8K conditional generation.
```shell
steps=32
p=0.6

# mdm
accelerate launch --multi_gpu --num_processes 8 evaluate_diff.py --tasks gsm8k --num_fewshot 0 --model mdlm --batch_size 1 --model_args model_name=1028,gen_length=256,ckpt_path=$mdm_path,steps=$steps,p=$p

# did
accelerate launch --multi_gpu --num_processes 8 evaluate_did.py --tasks gsm8k --num_fewshot 0 --model did --batch_size 1 --model_args model_name=1028,ckpt_path=$ckpt_path,steps=$steps,p=$p
```

## Acknowledgement
This repository is built upon [SMDM](https://github.com/ML-GSAI/SMDM).
