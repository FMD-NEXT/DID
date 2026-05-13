import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from noise_lib import add_noise_t, add_noise_lambda, add_noise_k
from torch.cuda.nvtx import range_push, range_pop, mark 


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

import time 

LOG_ZERO=-999999
def safe_log(x, ):
    return torch.where(x == 0, LOG_ZERO, torch.log(x))


def get_N_ratio(batch, remain_indices, seqlens, token_dim):
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

    
    # t1 = time.perf_counter()

    # prefix si_eq_tj and suffix si_eq_tj combined
    combined_eq = torch.stack([prefix_si_eq_tj_log, suffix_si_eq_tj_log_flipped], dim=-1).permute(1, 2, 3, 0) # (S, S, 2, B)

    # prefix dp and suffix dp combined, to save half kernel launches
    dtype = torch.float32
    combined_dp = torch.zeros(S+1, S+1, 2, B,  dtype=dtype, device=batch.device)  # (S+1, S+1, 2, B)
    combined_dp[:, 0] = 1
    combined_dp = safe_log(combined_dp)
    for i in range(1, S+1):
        prev = combined_dp[i-1]
        torch.logaddexp(prev[1:], combined_eq[i-1] + prev[:-1], out=combined_dp[i, 1:])

    prefix_dp, suffix_dp = combined_dp[:, :, 0].permute(2, 0, 1), combined_dp[:, :, 1].permute(2, 0, 1)
    suffix_dp = torch.flip(suffix_dp, [1, 2])


    # range_pop()
    # range_push('dp3')

    # N(y, x_0) prefix-suffix dp, where y = x_t.insert(s, v)
    V = token_dim 
    N_ratios = []
    for b in range(B):
        N = prefix_dp[b, -1, seqlens[b]]
        pr = prefix_dp[b, :-1, 1:seqlens[b] + 1]
        su = suffix_dp[b, 1:, S - seqlens[b] + 1:]
        # pr_su = (pr / N) * su   
        
        pr_su = (pr + su - N).exp() 
        # pr_su = (pr / N) * su  
        
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
        
    packed_N_ratios = torch.cat(N_ratios, 1) 

    # range_pop()
    return packed_N_ratios.t().coalesce() # (\sum_b |x_t|_b, V)


def select_sparse_non_pad(sparse_tensor, row_mask):
    """
    Select rows and exclude any elements in the first column.
    """
    indices = sparse_tensor.indices()
    values = sparse_tensor.values()
    
    # 获取坐标
    row_coords = indices[0]
    col_coords = indices[1]
    
    # 筛选条件：在选中的行中 AND 不在第一列(pad)
    selected_mask = row_mask[row_coords] & (col_coords != 0)
    
    selected_row_coords = row_coords[selected_mask]
    selected_col_coords = col_coords[selected_mask]
    selected_values = values[selected_mask]
    
    # 重新映射行索引
    cumulative_mapping = torch.cumsum(row_mask.int(), dim=0) - 1
    cumulative_mapping[~row_mask] = -1
    new_row_coords = cumulative_mapping[selected_row_coords]
    
    new_indices = torch.stack([new_row_coords, selected_col_coords])
    new_shape = (row_mask.sum().item(), sparse_tensor.shape[1])
    
    return torch.sparse_coo_tensor(new_indices, selected_values, new_shape).coalesce()
    

def get_loss_fn(noise, token_dim, train, sampling_eps=1e-3, loss_type='lambda_DCE',order = torch.arange(1024)):
    def DISE_loss(model, batch, cond = None):
        range_push('loss 1')
        t = (1 - sampling_eps) * Batch_Uniform_Sampler(batch.shape[0], type = 'vdm', device = batch.device) + sampling_eps
        sigma, dsigma = noise(t)
        sigma, dsigma = sigma[:,None], dsigma[:,None]
        esigm1 = torch.where(sigma < 0.5, torch.expm1(sigma),torch.exp(sigma) - 1 )

        # remove indices
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*batch.shape, device=batch.device) < move_chance
        move_indices[:, 0] = False # bos should not be removed
        remain_indices = ~move_indices

        packed_tokens = batch[remain_indices]
        seqlens = remain_indices.sum(-1)

        # N(y, x_0) / N(x_t, x_0) with pad
        t0 = time.perf_counter()
        packed_N_ratios = get_N_ratio(batch, remain_indices, seqlens, token_dim)
        t1 = time.perf_counter()

        # varlen: packed tokens, N ratios, seqlens w\o pad
        xt_non_pad_mask = (remain_indices & (batch != 0))[remain_indices] # pad id: 0
        seqlens = (remain_indices & (batch != 0)).sum(-1)
        packed_tokens = packed_tokens[xt_non_pad_mask]
        packed_N_ratios = select_sparse_non_pad(packed_N_ratios, xt_non_pad_mask)
        # ===

        nnz_indices = packed_N_ratios.indices()  # Shape (2, nnz), 2: b & s
        nnz_values = packed_N_ratios.values()  # Shape (nnz,)

        range_pop()

        # log time independent score
        range_push('loss 2')
        if train:
            model.train()
        else:
            model.eval()
        out = model(packed_tokens, seqlens, sigma.squeeze(-1)) 
        t2 = time.perf_counter()
        range_pop()
        
        range_push('loss 3')

        packed_dsigma = torch.repeat_interleave(dsigma, seqlens, dim=0)
        packed_esigm1 = torch.repeat_interleave(esigm1, seqlens, dim=0)
        coef = packed_dsigma / packed_esigm1 # (packed, 1)
        
        # loss, sparse computation
        with torch.amp.autocast("cuda", ):
            pos = (coef * out.exp()).sum() / batch.shape[0] 
            neg = - (coef[nnz_indices[0]].squeeze(-1) * (packed_N_ratios * out).coalesce().values()).sum() / batch.shape[0] 
            const = (coef[nnz_indices[0]].squeeze(-1) * nnz_values * (nnz_values.log() - 1)).sum() / batch.shape[0] 

            loss = pos + neg + const 
        range_pop()

        return loss, (
            pos, neg, const, t1-t0, t2-t1 
        )

    def t_DCE_loss(model, batch, cond = None):
        # sample t and add noise
        t = (1 - sampling_eps) * Batch_Uniform_Sampler(batch.shape[0], type = 'vdm', device = batch.device) + sampling_eps
        sigma, dsigma = noise(t)
        sigma, dsigma = sigma[:,None], dsigma[:,None]
        perturbed_batch = add_noise_t(batch, sigma, token_dim - 1)
        masked_index = perturbed_batch == token_dim - 1
        masked_batch = batch[masked_index]

        # compute c_theta and scaling factor
        if train:
            model.train()
        else:
            model.eval()
        log_condition = model(perturbed_batch)
        esigm1 = torch.where(sigma < 0.5, torch.expm1(sigma),torch.exp(sigma) - 1 )
        # compute score 
        log_condition -=esigm1.log()[...,None]

        # compute DCE loss
        loss = torch.zeros(*batch.shape, device=batch.device,dtype = log_condition.dtype)
        loss[masked_index] = - torch.gather(log_condition[masked_index], -1, masked_batch[..., None]).squeeze(-1)
        loss/= esigm1
        loss = (dsigma * loss).sum(dim=-1)
        return loss, (loss, loss, loss, 0,0)

    def lambda_DCE_loss(model, batch, cond = None):
        # sample lambda and add noise
        # Lambda = torch.rand(batch.shape[0], device=batch.device)
        Lambda = Batch_Uniform_Sampler(batch.shape[0], type = 'decoupled', device = batch.device)
        perturbed_batch = add_noise_lambda(batch, Lambda, token_dim - 1)
        masked_index = perturbed_batch == token_dim - 1
        masked_batch = batch[masked_index]
        
        if train:
            model.train()
        else:
            model.eval()
        log_condition = model(perturbed_batch)
        loss = torch.zeros(*batch.shape, device=batch.device,dtype = log_condition.dtype)
        loss[masked_index] = torch.gather(log_condition[masked_index], -1, masked_batch[..., None]).squeeze(-1)
        loss = - loss.sum(dim = -1).to(torch.float64)/Lambda.to(torch.float64)
        return loss, (loss, loss, loss, 0,0)

    def k_DCE_loss(model, batch, cond = None): # any-order ar loss
        # sample k and add noise
        k = torch.randint(1, batch.shape[1] + 1 ,(batch.shape[0],),device=batch.device)
        perturbed_batch = add_noise_k(batch, k, token_dim - 1)
        masked_index = perturbed_batch == token_dim - 1
        masked_batch = batch[masked_index]

        if train:
            model.train()
        else:
            model.eval()
        log_condition = model(perturbed_batch)
        loss = torch.zeros(*batch.shape, device=batch.device,dtype = log_condition.dtype)
        loss[masked_index] = torch.gather(log_condition[masked_index], -1, masked_batch[..., None]).squeeze(-1)
        loss = - loss.sum(dim = -1)/k * batch.shape[1]
        return loss.to(torch.float32), (loss, loss, loss, 0,0)
    
    def t_DSE_loss(model, batch, cond = None):
        # sample t and add noise
        t = (1 - sampling_eps) * Batch_Uniform_Sampler(batch.shape[0], type = 'vdm', device = batch.device) + sampling_eps
        sigma, dsigma = noise(t)
        sigma, dsigma = sigma[:,None], dsigma[:,None]
        perturbed_batch = add_noise_t(batch, sigma, token_dim - 1)
        masked_index = perturbed_batch == token_dim - 1
        masked_batch = batch[masked_index]

        # compute c_theta and scaling factor
        if train:
            model.train()
        else:
            model.eval()
        log_condition = model(perturbed_batch)
        esigm1 = torch.where(sigma < 0.5, torch.expm1(sigma),torch.exp(sigma) - 1 )
        # compute score (reuse log_condition to save memory)
        log_condition -=esigm1.log()[...,None]

        scaling_factor = 1 / esigm1.expand_as(perturbed_batch)
        
        # compute three terms
        loss = torch.zeros(*batch.shape, device=batch.device,dtype = log_condition.dtype)
        # add negative term
        loss[masked_index] = - torch.gather(log_condition[masked_index], -1, masked_batch[..., None]).squeeze(-1)
        loss/= esigm1
        # add pos term
        loss[masked_index] += log_condition[masked_index][:, :-1].exp().sum(dim=-1)

        # add const term 
        loss[masked_index] += scaling_factor[masked_index] * (scaling_factor[masked_index].log() - 1)
        l =  (dsigma * loss).sum(dim=-1)
        return l, (l,l,l, 0,0)
        
    if loss_type == 'DISE':
        return DISE_loss
        
    elif loss_type == 't_DCE':
        return t_DCE_loss
    elif loss_type == 't_DSE':
        return t_DSE_loss
    elif loss_type == 'k_DCE':
        return k_DCE_loss
    elif loss_type =='lambda_DCE':
        return lambda_DCE_loss
    else:
        raise NotImplementedError(f'Loss type {loss_type} not supported yet!')


def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'SGD':
        optimizer = optim.SGD(params, lr=config.optim.lr, momentum=config.optim.beta1, weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, 
                    scaler, 
                    params, 
                    step, 
                    lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip,
                    total=config.training.n_iters,
                    cos_decay=config.cos_decay):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        scaler.unscale_(optimizer)

        warmup_ratio = np.minimum(step / warmup, 1.0) if warmup > 0 else 1
        cos_decay_ratio = 0.5 * (1 + np.cos(np.pi * np.maximum((step - warmup) / (total - warmup), 0))) if cos_decay else 1

        for g in optimizer.param_groups:
            g['lr'] = lr * warmup_ratio * cos_decay_ratio

        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip) 
            # torch.nn.utils.clip_grad_value_(params, clip_value=grad_clip)

        scaler.step(optimizer)
        scaler.update()

    return optimize_fn


def get_step_fn(noise, token_dim,  train, optimize_fn, accum, loss_type):
    loss_fn = get_loss_fn(noise, token_dim, train, loss_type = loss_type)

    accum_iter = 0
    total_loss = 0

    total_l1 = 0
    total_l2 = 0
    total_c = 0

    total_d1 = 0
    total_d2 = 0

    def step_fn(state, batch, cond=None):
        nonlocal accum_iter 
        nonlocal total_loss
        
        nonlocal total_l1
        nonlocal total_l2
        nonlocal total_c

        nonlocal total_d1
        nonlocal total_d2

        model = state['model']

        grad_sparsity, grad_norm_l2, weight_norm = 0, 0, 0
        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']
            range_push('loss')
            l, (l1, l2, c, d1, d2) = loss_fn(model, batch, cond=cond)
            loss = l.mean() / accum

            l1, l2, c = l1.mean() / accum, l2.mean() / accum, c.mean() / accum
            mark('fwd/bwd')
            scaler.scale(loss).backward()
            range_pop()

            accum_iter += 1
            total_loss += loss.detach()

            total_l1 += l1.detach()
            total_l2 += l2.detach()
            total_c += c.detach()

            total_d1 += d1 
            total_d2 += d2 

            if accum_iter == accum:
                accum_iter = 0

                state['step'] += 1
                range_push('optim')
    
                def get_grad_sparsity(model, th=1e-4):
                    total = torch.tensor(0.0, dtype=torch.double).to(next(model.parameters()).device)
                    total1 = torch.tensor(0.0, dtype=torch.double).to(next(model.parameters()).device)
                    total2 = torch.tensor(0.0, dtype=torch.double).to(next(model.parameters()).device)
                    num_params = torch.tensor(0.0, dtype=torch.double).to(next(model.parameters()).device)
                    
                    for p in model.parameters():
                        total += (p.grad.detach().data.abs() < th).double().sum()
                        total1 += (p.grad.detach().data.abs()).double().sum()
                        total2 += (p.grad.detach().data ** 2).double().sum()
                        num_params += p.numel()
                    
                    # return total / num_params
                    return total1 / total2.sqrt() / num_params.sqrt()

                def get_grad_norm_L2_avg(model):
                    total_norm = torch.tensor(0.0, dtype=torch.double).to(next(model.parameters()).device)
                    num_params = torch.tensor(0.0, dtype=torch.double).to(next(model.parameters()).device)
                    
                    for p in model.parameters():
                        total_norm += (p.grad.detach().data ** 2).sum()
                        num_params += p.numel()
                    
                    return torch.sqrt(total_norm / num_params)
        
                def get_global_avg_weight_norm(model):
                    total_norm = torch.tensor(0.0, dtype=torch.double).to(next(model.parameters()).device)
                    num_params = torch.tensor(0.0, dtype=torch.double).to(next(model.parameters()).device)
                    
                    for p in model.parameters():
                        total_norm += (p.detach().data ** 2).sum()
                        num_params += p.numel()
                    
                    return torch.sqrt(total_norm / num_params)

        
                grad_norm_l2 = get_grad_norm_L2_avg(model)
                grad_sparsity = get_grad_sparsity(model)
                weight_norm = get_global_avg_weight_norm(model)

                optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])
                state['ema'].update(model.parameters())


                optimizer.zero_grad()
                range_pop()
                
                loss = total_loss
                total_loss = 0

                l1, l2, c = total_l1, total_l2, total_c
                total_l1, total_l2, total_c = 0, 0, 0

                d1, d2 = total_d1, total_d2 
                total_d1, total_d2 = 0 ,0
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                l, (l1, l2, c, d1, d2 ) = loss_fn(model, batch, cond=cond)
                loss = l.mean() 
                ema.restore(model.parameters())


        return loss, (l1, l2, c), (grad_sparsity, grad_norm_l2, weight_norm), (d1, d2)

    return step_fn
