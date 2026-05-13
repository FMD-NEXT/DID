'''
This file is inspired by the code provided by the author of https://arxiv.org/abs/2406.11473
'''
import torch
import re
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from transformers import AutoTokenizer
from lit_gpt.diffmodel import TransEncoder_Wt as TransEncoder, Config
from safetensors.torch import load_file

from datetime import timedelta
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
    find_executable_batch_size,
)

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



import abc
import torch
import torch.nn as nn
import time 
from itertools import chain 

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

    t0 = time.perf_counter()
    
    # prefix si_eq_tj and suffix si_eq_tj combined
    combined_eq = torch.stack([prefix_si_eq_tj_log, suffix_si_eq_tj_log_flipped], dim=-1).permute(1, 2, 3, 0)

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

    t1 = time.perf_counter()

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

    t2 = time.perf_counter()
    return ret, (t1-t0, t2-t1)

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

def DISE_loss(model, batch, remove_indices, t, token_dim = 32000, pad_id = PAD_ID, cond = None, train=True, sparse=True, per_token_loss=False):
    
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

    return loss, (pos, neg, const, packed_N_ratios.sum() + seqlens.sum() - num_tokens, t1, t2)

def get_where_to_compute(seqlens, prompt_len, removed_indices):
    batch_arange = torch.arange(seqlens.max(), device=seqlens.device).repeat(seqlens.shape[0], 1) # (b, max(s))
    packed_arange = batch_arange[batch_arange < seqlens[:, None]] # (\sum_b |x_t|_b, )
    where_to_compute = (packed_arange >= prompt_len - 1).unsqueeze(-1) # (\sum_b |x_t|_b, 1)

    # cumsum = torch.cumsum(seqlens, dim=0) # (b, )
    # where_to_compute[cumsum - 1] = removed_indices[:, -1].unsqueeze(-1) # if the last element in the target part is not removed, we don't compute the insertion loss after this element

    return where_to_compute 

@register_model("did")
class MDLMEvalHarness(LM):
    def __init__(
            self,
            model_name="tiny",
            ckpt_path=None,
            mask_id=32000,
            max_length=2048,
            batch_size=32,
            mc_num=1024,
            padding=False,
            nll_type='mc',
            greddy=False,
            cfg=0.,
            device="cuda",

            # did configs
            per_token_loss=True,
            cond_nll=False,
            arch='Wt', # Wt, GAda, Wot
            
            # gen
            temp=0.1,
            steps=1024,
            p=0.9
    ):
        super().__init__()
        assert nll_type in ['mc', 'chain_rule']
        
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
            
        if accelerator.num_processes > 1:
            self.device = self.accelerator.device
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.device = torch.device(device)
            self._rank = 0
            self._world_size = 1

        # model_name = f'Diff_LLaMA_Wt_{model_name}M'
        # model_name = f'Diff_LLaMA_Wot_{model_name}M'
        model_name = f'Diff_LLaMA_{arch}_{model_name}M'
        config = Config.from_name(model_name)
        self.model = TransEncoder(config).to(device)

        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device)['model'])
        # self.model.load_state_dict(torch.load(ckpt_path))
        self.model.eval()


        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained('tinyllama_tokenizer')

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.padding = padding
        self.nll_type = nll_type
        self.greddy = greddy

        self.cond_nll = cond_nll
        self.per_token_loss = per_token_loss

        self.cfg = cfg
        self.temp = temp
        self.steps = steps

        # did sampler
        self.eps = 1e-5
        self.BOS = 1
        self.token_dim = 32000
        self.p = p


    @torch.no_grad()
    def _eval_target_nll_mc(self, prefix, target):
        '''
        Employ Monte Carlo estimation to establish a lower bound of the log-likelihood
        '''
        # print(f'{len(prefix), len(target) = }');exit()
        # print(f'{prefix.tolist(), target.tolist() = }');exit()
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device) # repeat for parallel mc evaluation

        prompt_index = (torch.arange(seq.shape[1], device=self.device) < len(prefix)).unsqueeze(0).repeat(seq.shape[0], 1) # (b, s)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            _, removed_indices, t = forward_process_did(seq)

            if self.cond_nll:
                removed_indices[prompt_index] = False

                seqlens = (~removed_indices).sum(-1)
                prompt_len = len(prefix)

                where_to_compute = get_where_to_compute(seqlens, prompt_len, removed_indices)
            else:
                where_to_compute = None
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss, _ = DISE_loss(self.model, seq, removed_indices, t, train=False, cond=where_to_compute, per_token_loss=self.per_token_loss)

            loss_acc.append(loss.cpu())

        return sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        return False 


    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 2048

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                if self.nll_type == 'mc':
                    ll = -self._eval_target_nll_mc(prefix, target)
                else:
                    raise NotImplementedError(self.nll_type)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        return out

    def loglikelihood_rolling(self, requests: list[Instance]):
        raise NotImplementedError

    
    
    def generate_until(self, requests: list[Instance]):
        def _tokenize(e):
            return {
                "question": self.tokenizer(e["question"])["input_ids"],
                "question_text": e["question"],
                "until": e["until"],
            }

        ds = [{"question": req.args[0], "until": req.args[1]['until']} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        
        batch_size = self.batch_size 
        length = len(ds)
        iters = length // batch_size if length % batch_size == 0 else length // batch_size + 1

        out = []
        for i in tqdm(range(iters), desc="Generating..."):
            end_index = (i + 1) * batch_size if (i + 1) * batch_size < length else length
            data = ds[i * batch_size: end_index]
            prompt = [q for q in data["question"]]

            stop_tokens = data["until"][0]
            
            generated_answer = diff_sample(self.model,
                             self.tokenizer,
                             prompt,
                             steps=self.steps,
                             context_length=256,
                             device=self.device,
                             topp=self.p)

            generated_answer = [ans[len(prompt[i]):] for i, ans in enumerate(generated_answer)]

            generated_answer = self.tokenizer.batch_decode(generated_answer, skip_special_tokens=False)

            for i in range(len(generated_answer)):
                for stop_seq in stop_tokens:
                    if stop_seq in generated_answer[i]:
                        generated_answer[i] = generated_answer[i].split(stop_seq)[0]

            # remove special tokens
            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.batch_decode(generated_answer_ids, skip_special_tokens=True)

            out += generated_answer

            self.accelerator.wait_for_everyone()


        return out

    

def sample_categorical(categorical_probs, temperature=1.):
    gumbel_norm = (-torch.rand_like(categorical_probs).log()) ** temperature
    return (categorical_probs / gumbel_norm).argmax(dim=-1)

def top_p_sampling(probs, p=0.9):

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
    probs.masked_fill_(indices_to_remove, 0)
    probs /= probs.sum(dim=-1, keepdim=True)
    index = sample_categorical(probs.to(torch.float64))
    
    return index


def get_update_rate(t, steps):
    eps = 1e-5
    dt = (1 - eps) / steps
    curr_sigma, next_sigma = noise(t)[0], noise(t - dt)[0]
    d_curr_sigma = noise(t)[1]
    
    update_rate = dt * d_curr_sigma * (-curr_sigma).exp() / (1 - (-curr_sigma).exp())
    return update_rate

def get_sigma(t):
    curr_sigma = noise(t)[0]
    return curr_sigma



@ torch.no_grad()
def diff_sample(model, tokenizer, prompt, steps=512, context_length=256, eps=1e-5, dim=32000, device='cuda', topp=0.9):
    batch_size = len(prompt)
    max_len = context_length * 2

    x = prompt 
    packed_x_tensor =  torch.tensor(list(chain(*x)), dtype=torch.int32).to(device)
    seqlens = torch.tensor([len(xi) for xi in x] , dtype=torch.int32).to(device)
    seqlens_init = seqlens.clone() 

    unpacked_score = torch.zeros((batch_size, max_len, dim + 1), dtype=torch.float64).to(device) # (B, max_len, V + 1)

    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    for i in range(steps):
        t = timesteps[i]
        sigma = get_sigma(t).repeat(batch_size)
        update_rate = get_update_rate(t, steps) if i < steps - 1 else 1 + 1e-3

        seqid = torch.repeat_interleave(torch.arange(batch_size).to(device), seqlens)

        # unpacked x tensor 
        unpack_mask = torch.arange(max_len, device=device)[None, :] < seqlens[:, None]  # (B, max_len)
        rows, cols = torch.where(unpack_mask)  # (B, max_len)

        # NFE
        with torch.amp.autocast("cuda", torch.bfloat16):
            out = model(packed_x_tensor, seqlens=seqlens, c=sigma)
        unpacked_score[rows, cols, :-1] = out.double().exp() 
        
        packed_score = unpacked_score[unpack_mask]

        probs = packed_score * update_rate
        VOID = dim

        # location sampling
        insertions = torch.zeros_like(packed_x_tensor).int()

        prob_void = 1 - probs.sum(-1, keepdim=False)
        void_locations = torch.rand_like(prob_void) < prob_void
        insertions[void_locations] = VOID 

        # topp
        probs[~void_locations] /= probs[~void_locations].sum(-1, keepdim=True)

        insertions[~void_locations] = top_p_sampling(probs[~void_locations], p=topp).int()
            
        
        # mask prefix
        seq_mask = torch.arange(max_len, device=device)[None, :] < seqlens[:, None] # (b, s)
        prompt_mask = torch.arange(max_len, device=device)[None, :] < seqlens_init[:, None] - 1 # (b, s)
        cond = prompt_mask[seq_mask] # (\sum_b |x_t|_b, )

        insertions[cond] = VOID


        # update seqlens
        seqlens.scatter_add_(0, seqid, (insertions != VOID).int())

        # update x 
        inserted_with_void = torch.stack((packed_x_tensor, insertions), dim=1).view(-1) # (2 * packed, )
        packed_x_tensor = inserted_with_void[inserted_with_void != VOID]


    res = [_x.tolist() for _x in torch.split(packed_x_tensor, seqlens.tolist())]
    return res # [_x[1:] for _x in res] 


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()