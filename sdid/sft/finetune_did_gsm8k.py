
#  torchrun --nproc_per_node=8 sft/finetune_did_gsm8k.py --model 1028 --pretrain_path workdir/scaling_debug/mdm-1028M-3300.0/iter-3959392-ckpt.pth 

import glob
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union
import math
import lightning as L
import torch
import torch.nn.functional as F 
from lightning.fabric.strategies import FSDPStrategy, XLAStrategy, DDPStrategy
from torch.utils.data import DataLoader
from functools import partial
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
# from apex.optimizers import FusedAdam #torch optimizer has a cuda backend, which is faster actually
from lit_gpt.config import Config
from lit_gpt.diffmodel import Block_Wt as Block, SelfAttention_VarLen as SelfAttention, TransEncoder_Wt as TransEncoder
from lit_gpt.packed_dataset import CombinedDataset, PackedDataset
from lit_gpt.speed_monitor import SpeedMonitorFabric as Monitor
from lit_gpt.speed_monitor import estimate_flops, measure_flops
from lit_gpt.utils import chunked_cross_entropy, get_default_supported_precision, num_parameters, step_csv_logger, lazy_load
from pytorch_lightning.loggers import WandbLogger
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from gsm8k_data import preprocess_gsm8k
from transformers import AutoTokenizer
import random
import argparse
from safetensors.torch import load_file

from torch.cuda.nvtx import range_push, range_pop

def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--model', type=int, help='model parameters')
    parse.add_argument('--bs', type=int, default=512, help='batch size')
    parse.add_argument('--beta2', type=float, default=0.95, help='beta2')
    parse.add_argument('--lr', type=float, default=2e-4, help='lr')
    parse.add_argument('--loss', type=str, default='tok', help='tok or seq')
    parse.add_argument('--epoch', type=int, default=40, help='training epoch')
    parse.add_argument('--pretrain_path', type=str, help='pretrain ckpt path')
    parse.add_argument('--nodes_num', type=int, default=1, help='number of devices')
    args = parse.parse_args()
    return args

args = parse_args()
# model_name = f'Diff_LLaMA_{args.model}M'  # config
model_name = f'Diff_LLaMA_Wt_{args.model}M'  # config
out_dir = Path('workdir')

# Hyperparameters
num_of_devices = 8
global_batch_size = int(args.bs / args.nodes_num)
learning_rate = args.lr
micro_batch_size = int(args.bs / num_of_devices)
max_step = int(769240 * args.epoch / args.bs)
warmup_steps = int(max_step * 0.01) # 0.1, 0.01
log_step_interval = 10
save_step_interval = max_step // 20

weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
# beta2 = args.beta2
grad_clip = 1.0
decay_lr = True
min_lr = learning_rate / 10

batch_size = global_batch_size // num_of_devices
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
warmup_iters = warmup_steps * gradient_accumulation_steps




max_iters = max_step * gradient_accumulation_steps
lr_decay_iters = max_iters
log_iter_interval = log_step_interval * gradient_accumulation_steps


hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
logger = step_csv_logger("out", model_name, flush_logs_every_n_steps=log_iter_interval)

### DID
import abc
import torch
import torch.nn as nn

def Batch_Uniform_Sampler(B, type = 'naive', device = 'cuda'):
    def vdm_sampler(B, device):
        u_0 = torch.rand(1, device=device)  # Sample u_0 from U(0, 1)
        t = [(u_0 + i / B) % 1 for i in range(B)]
        t = torch.tensor(t, device=device)
        return t
    
    def decoupled_sampler(B, device):
        u = torch.rand(B, device=device)  # Sample B independent values from U(0, 1)
        t = [(u[i] + i) / B for i in range(B)]
        t = torch.tensor(t, device=device)
        return t
    if type == 'naive':
        return torch.rand(B, device = device)
    elif type == 'vdm':
        return vdm_sampler(B, device)
    elif type == 'decoupled':
        return decoupled_sampler(B, device)
    else:
        raise ValueError(f"{type} not valid")

class Noise(abc.ABC, nn.Module):
    """
    Baseline forward method to get the total + rate of noise at a timestep
    """
    def forward(self, t):
        return self.total_noise(t), self.rate_noise(t)

    """
    Assume time goes from 0 to 1
    """
    @abc.abstractmethod
    def rate_noise(self, t):
        """
        Rate of change of noise ie g(t)
        """
        pass

    @abc.abstractmethod
    def total_noise(self, t):
        """
        Total noise ie \\int_0^t g(t) dt + g(0)
        """
        pass

class LogLinearNoise(Noise, nn.Module):
    """
    Log Linear noise schedule built so that 1 - 1/e^(n(t)) interpolates between 0 and ~1
    when t goes from 0 to 1. Used for absorbing

    Total noise is -log(1 - (1 - eps) * t), so the sigma will be (1 - eps) * t
    """
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.empty = nn.Parameter(torch.tensor(0.0))

    def rate_noise(self, t):
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t):
        return -torch.log1p(-(1 - self.eps) * t)

noise = LogLinearNoise() 

def forward_process_did(batch, sampling_eps=1e-3):
    b, s = batch.shape
    t = (1 - sampling_eps) * Batch_Uniform_Sampler(batch.shape[0], type = 'vdm', device = batch.device) + sampling_eps
    remove_indices = torch.rand((b, s), device=batch.device) < t[:, None]
    remove_indices[:, 0] = False # bos should not be removed
    return batch, remove_indices, t # (b,s), (b,s), (b,)

LOG_ZERO=-999999
def safe_log(x, ):
    return torch.where(x == 0, LOG_ZERO, torch.log(x))

def get_N_ratio_logdomain(batch, remain_indices, seqlens, token_dim, sparse=True):
    t0 = time.perf_counter()
    # batched data alignment
    prefix_padded_xt = torch.zeros_like(batch, device=batch.device) - 1 # init as -1
    prefix_data_mask = seqlens[..., None] > torch.arange(batch.shape[1], device=batch.device)[None, ...]
    prefix_padded_xt[prefix_data_mask] = batch[remain_indices]
    prefix_si_eq_tj = (batch.unsqueeze(-1) == prefix_padded_xt.unsqueeze(-2))#.double()
    prefix_si_eq_tj_log = torch.log(prefix_si_eq_tj)

    suffix_padded_xt = torch.zeros_like(batch, device=batch.device) - 1 # init as -1
    suffix_data_mask = seqlens[..., None] > torch.arange(batch.shape[1] - 1, -1, -1, device=batch.device)[None, ...]
    suffix_padded_xt[suffix_data_mask] = batch[remain_indices]

    suffix_si_eq_tj_flipped = (torch.flip(batch, [1]).unsqueeze(-1) == torch.flip(suffix_padded_xt, [1]).unsqueeze(-2))
    suffix_si_eq_tj_log_flipped = torch.log(suffix_si_eq_tj_flipped)

    B, S = batch.shape

    
    t1 = time.perf_counter()

    # prefix si_eq_tj and suffix si_eq_tj combined
    combined_eq = torch.stack([prefix_si_eq_tj_log, suffix_si_eq_tj_log_flipped], dim=-1).permute(1, 2, 3, 0) # (S, S, 2, B)

    # prefix dp and suffix dp combined, to save half kernel launches
    dtype = torch.float64
    combined_dp = torch.zeros(S+1, S+1, 2, B,  dtype=dtype, device=batch.device)  # (S+1, S+1, 2, B)
    combined_dp[:, 0] = 1
    combined_dp = safe_log(combined_dp)
    for i in range(1, S+1):
        prev = combined_dp[i-1]
        torch.logaddexp(prev[1:], combined_eq[i-1] + prev[:-1], out=combined_dp[i, 1:])

    prefix_dp, suffix_dp = combined_dp[:, :, 0].permute(2, 0, 1), combined_dp[:, :, 1].permute(2, 0, 1)
    suffix_dp = torch.flip(suffix_dp, [1, 2])


    # =========================================================================================
    t2 = time.perf_counter()

    # N(Ins(x_t, i, v), x_0) / N(x_t, x_0) prefix-suffix dp
    V = token_dim 
    N_ratios = []

    for b in range(B):
        N = prefix_dp[b, -1, seqlens[b]] 
        pr = prefix_dp[b, :-1, 1:seqlens[b] + 1]
        su = suffix_dp[b, 1:, S - seqlens[b] + 1:]
        
        pr_su = (pr + su - N).exp() 
        # pr_su = (pr / N) * su  

        if sparse: # sparse
            S, T = pr_su.shape
            rows = batch[b].unsqueeze(1).expand(S, T).reshape(-1)
            cols = torch.arange(T).to(batch.device).unsqueeze(0).expand(S, T).reshape(-1)
            values = pr_su.reshape(-1)

            mask = values.abs() >= 1e-6  # (S*T,) bool mask
            rows = rows[mask]
            cols = cols[mask]
            values = values[mask]

            indices = torch.stack([rows, cols], dim=0)
            N_ratio = torch.sparse_coo_tensor(indices, values, size=(V, T))
            N_ratios.append(N_ratio)
        else: # dense
            N_ratio = torch.zeros((V, pr_su.size(1)), dtype=pr_su.dtype, device=batch.device) # (V, T)
            N_ratio.index_add_(0, batch[b], pr_su)
            N_ratios.append(N_ratio)

    packed_N_ratios = torch.cat(N_ratios, 1) 

    if sparse: # sparse
        ret = packed_N_ratios.t().coalesce() # (\sum_b |x_t|_b, V)
    else: # dense
        ret = packed_N_ratios.t() # (\sum_b |x_t|_b, V)

    t3 = time.perf_counter()
    return ret, (t3-t0, t2-t1)

def select_sparse_non_pad(sparse_tensor, row_mask, pad_id=2):
    """
    Select rows and exclude any elements in the pad column.
    """
    indices = sparse_tensor.indices()
    row_coords = indices[0]
    col_coords = indices[1]
    values = sparse_tensor.values()
    
    selected_mask = row_mask[row_coords] & (col_coords != pad_id)
    
    selected_row_coords = row_coords[selected_mask]
    selected_col_coords = col_coords[selected_mask]
    selected_values = values[selected_mask]
    
    # new coords
    cumulative_mapping = torch.cumsum(row_mask.int(), dim=0) - 1
    cumulative_mapping[~row_mask] = -1
    new_row_coords = cumulative_mapping[selected_row_coords]
    new_indices = torch.stack([new_row_coords, selected_col_coords])

    new_shape = (row_mask.sum().item(), sparse_tensor.shape[1])
    
    return torch.sparse_coo_tensor(new_indices, selected_values, new_shape).coalesce()
    
PAD_ID = 2

def DISE_loss(model, batch, remove_indices, t, token_dim = 32000, pad_id = PAD_ID, cond = None, train=True, sparse=True, per_token_loss=True):
    
    sigma, dsigma = noise(t)
    sigma, dsigma = sigma[:,None], dsigma[:,None]
    esigm1 = torch.where(sigma < 0.5, torch.expm1(sigma),torch.exp(sigma) - 1 )

    remain_indices = ~remove_indices

    packed_tokens = batch[remain_indices]
    seqlens = remain_indices.sum(-1) # seqlens with pad

    # N(y, x_0) / N(x_t, x_0) with pad
    packed_N_ratios, (t1, t2) = get_N_ratio_logdomain(batch, remain_indices, seqlens, token_dim, sparse=sparse)

    # varlen: packed tokens, N ratios, seqlens without pad
    xt_non_pad_mask = (remain_indices & (batch != pad_id))[remain_indices] 
    seqlens = (remain_indices & (batch != pad_id)).sum(-1) # seqlens without pad
    packed_tokens = packed_tokens[xt_non_pad_mask]
    
    if sparse: # sparse
        packed_N_ratios = select_sparse_non_pad(packed_N_ratios, xt_non_pad_mask, pad_id)
    else: # dense
        packed_N_ratios = packed_N_ratios[xt_non_pad_mask]
        packed_N_ratios[:, pad_id] = 0

    # coef: 1/t
    packed_dsigma = torch.repeat_interleave(dsigma, seqlens, dim=0)
    packed_esigm1 = torch.repeat_interleave(esigm1, seqlens, dim=0)
    coef = packed_dsigma / packed_esigm1 # (packed, 1)

    
    if cond is not None:
        coef *= cond
        prompt_tokens = (~cond).sum()
    else:
        prompt_tokens = 0

    # model output: log(scaled score)
    num_tokens = (batch != pad_id).sum() - prompt_tokens
    num_seqs = batch.shape[0]
    D = num_tokens if per_token_loss else num_seqs
    
    if train:
        model.train()
    else:
        model.eval()
            
    out = model(packed_tokens, sigma.squeeze(-1), seqlens) 

    # loss 
    with torch.amp.autocast("cuda", ):
        pos = (coef * out.exp()).sum() / D   
        if sparse: # sparse
            nnz_indices = packed_N_ratios.indices()  # Shape (2, nnz), 2: b & s
            nnz_values = packed_N_ratios.values()  # Shape (nnz,)
            neg = - (coef[nnz_indices[0]].squeeze(-1) * (packed_N_ratios * out).coalesce().values()).sum() / D
            const = (coef[nnz_indices[0]].squeeze(-1) * nnz_values * (nnz_values.log() - 1)).sum() / D
        else: # dense
            neg = - (coef * packed_N_ratios * out ).sum() / D    
            const = (coef * packed_N_ratios * (safe_log(packed_N_ratios) - 1)).sum() / D  

    loss = pos + neg + const 

    return loss, (pos, neg, const, packed_N_ratios.sum() + seqlens.sum() - num_tokens - prompt_tokens, t1, t2)



def extract_number(filename):
    match = re.search(r'iter-(\d+)-ckpt\.pth', str(filename))
    return int(match.group(1)) if match else 0


def setup(
    devices: int = 8,
    precision: Optional[str] = None,
    tpu: bool = False,
    resume: Union[bool, Path] = True,
) -> None:
    global out_dir
    hp_name = f'did-bs-{args.bs}-beta2{args.beta2}-lr{args.lr}-loss{args.loss}-epoch{args.epoch}-gsm8k-{args.model}M'
    out_dir = Path('workdir/finetune') / hp_name
    pretrain_path = args.pretrain_path
    wandb_logger = WandbLogger(name=hp_name, save_dir=out_dir, project='scaling', offline=True)

    precision = precision or get_default_supported_precision(training=True, tpu=tpu)

    if devices > 1:
        if tpu:
            # For multi-host TPU training, the device count for Fabric is limited to the count on a single host.
            devices = "auto"
            strategy = XLAStrategy(sync_module_states=False)
        else:
            # strategy = FSDPStrategy(
            #     auto_wrap_policy={Block},
            #     activation_checkpointing_policy=None,
            #     state_dict_type="full",
            #     limit_all_gathers=True,
            #     cpu_offload=False,
            # )
            strategy = DDPStrategy()
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=[logger, wandb_logger])
    fabric.print(hparams)
    fabric.launch(main, pretrain_path, resume)
    # main(fabric, pretrain_path, resume)



def main(fabric, pretrain_path, resume):
    monitor = Monitor(fabric, window_size=2, time_unit="seconds", log_iter_interval=log_iter_interval)

    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    config = Config.from_name(model_name)

    tokenizer = AutoTokenizer.from_pretrained('tinyllama_tokenizer',
                                            padding_side="right", use_fast=True)

    train_set = preprocess_gsm8k(tokenizer, max_length=256)

    fabric.seed_everything(42)  # same seed for every process to init model (FSDP)
    train_dataloader = DataLoader(train_set, batch_size=micro_batch_size, shuffle=True, drop_last=True,
                                    num_workers=8, pin_memory=True, persistent_workers=True)
    train_dataloader = fabric.setup_dataloaders(train_dataloader)

    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=False):
        model = TransEncoder(config)
        model.apply(partial(model._init_weights ,n_layer=config.n_layer))

        
        model.load_state_dict(torch.load(pretrain_path, weights_only=True)['model'])
        # model.load_state_dict(torch.load(pretrain_path, map_location=fabric.global_rank)['model'])

        fabric.print(f"Loading model from {pretrain_path}")

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")

    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False
    )
    # optimizer = FusedAdam(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2),adam_w_mode=True)
    optimizer = fabric.setup_optimizers(optimizer)

    state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}

    if resume is True:
        try:
            resume = sorted(out_dir.glob("*.pth"), key=extract_number)[-1]
        except:
            resume = False
    if resume :
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, monitor, resume)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

PAD_ID = 2


def train(fabric, state, train_dataloader, monitor, resume):
        model = state["model"]
        optimizer = state["optimizer"]

        with torch.device("meta"):
            meta_model = TransEncoder(model.config)
            # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
            # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
            # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
            estimated_flops = estimate_flops(meta_model) * micro_batch_size
            fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
            x = torch.randint(0, 1, (micro_batch_size, model.config.block_size))
            # measured_flos run in meta. Will trigger fusedRMSNorm error
            #measured_flops = measure_flops(meta_model, x)
            #fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
            del meta_model, x

        total_lengths = 0
        total_t0 = time.perf_counter()

        if fabric.device.type == "xla":
            import torch_xla.core.xla_model as xm

            xm.mark_step()
        
        
        initial_iter = state["iter_num"]
        curr_iter = 0

        def get_train_dataloader(dataset_loader):
            while True:
                for data in dataset_loader:
                    yield data
        train_dataloader_ = get_train_dataloader(train_dataloader)
                
        # loss_func = CrossEntropyLoss(reduction='none')
        for train_data in train_dataloader_:
            # resume loader state. This is not elegant but it works. Should rewrite it in the future.
            if resume:
                if curr_iter < initial_iter:
                    curr_iter += 1
                    continue
                else:
                    resume = False
                    curr_iter = -1
                    fabric.barrier()
                    fabric.print("resume finished, taken {} seconds".format(time.perf_counter() - total_t0))
            if state["iter_num"] >= max_iters:
                break
            
            # determine and set the learning rate for this iteration
            lr = get_lr(state["iter_num"]) if decay_lr else learning_rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            iter_t0 = time.perf_counter()
            input_ids = train_data['data'] # [prompt + answer + padding], length=2048
            prompt_length = train_data['input_length']  # prompt length
            max_length = 256
            input_ids = input_ids[:, :max_length] # (B, S)

            input_ids, removed_indices, t = forward_process_did(input_ids)

            # prompt should not be noised
            prompt_index = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :] < prompt_length[:, None] # (b, s)
            removed_indices[prompt_index] = False
            
            # cond mask     
            noised_seqlens = ((~removed_indices) & (input_ids != PAD_ID)).sum(-1)

            seq_mask = torch.arange(noised_seqlens.max(), device=noised_seqlens.device)[None, :] < noised_seqlens[:, None] # (b, s)
            prompt_mask = torch.arange(noised_seqlens.max(), device=noised_seqlens.device)[None, :] >= prompt_length[:, None] - 1 # (b, s)

            cond = prompt_mask[seq_mask].unsqueeze(-1) # (\sum_b |x_t|_b, 1)

            # cond = None 

            is_accumulating = (state["iter_num"] + 1) % gradient_accumulation_steps != 0
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                

                loss, (l1,l2,l3,r1,r2,r3) = DISE_loss(model, input_ids, removed_indices, t, token_dim=32000, pad_id=2, cond=cond, per_token_loss=args.loss == 'tok', train=True)  
                (l1,l2,l3,r1) = (l1.item(),l2.item(),l3.item(),r1.item())


                fabric.backward(loss / gradient_accumulation_steps)

            if not is_accumulating:
                fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                state["step_count"] += 1
            elif fabric.device.type == "xla":
                xm.mark_step()
            state["iter_num"] += 1
            # input_id: B L 
            total_lengths += input_ids.size(1)
            t1 = time.perf_counter()
            fabric.print(
                    f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, ({l1:.4f}, {l2:.4f}, {l3:.4f}, {r1:.4f}, {r2 * 1000:.4f}ms, {r3 * 1000:.4f}ms,) iter time:"
                    f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
                    f" remaining time: {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600:.2f} hours. " 
                    # print days as well
                    f" or {(t1 - total_t0) / (state['iter_num'] - initial_iter) * (max_iters - state['iter_num']) / 3600 / 24:.2f} days. "
                )
    
            monitor.on_train_batch_end(
                state["iter_num"] * micro_batch_size,
                t1 - total_t0,
                # this assumes that device FLOPs are the same and that all devices have the same batch size
                fabric.world_size,
                state["step_count"],
                flops_per_batch=estimated_flops,
                lengths=total_lengths,
                train_loss = loss.item()
            )

            if not is_accumulating and (state["step_count"] % save_step_interval == 0 or state["step_count"] == max_step):
                checkpoint_path = out_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
                fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
                fabric.save(checkpoint_path, state)




# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    setup()
