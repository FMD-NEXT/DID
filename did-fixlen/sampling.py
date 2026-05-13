import abc
import torch
import torch.nn.functional as F
from catsample import sample_with_strategy, sample_categorical


import abc
from copy import deepcopy
from itertools import chain 
from torch.cuda.nvtx import range_push, range_pop, mark 


class Sampler(abc.ABC):
    def __init__(self, model, batch_dims, token_dim, strategy, strategy_para=None, device=torch.device('cuda')):
        super().__init__()
        self.model = model
        self.batch_dims = batch_dims
        self.device = device
        self.strategy = strategy
        self.strategy_para = strategy_para
        self.token_dim = token_dim

    @abc.abstractmethod
    def sample(self, steps):
        raise NotImplementedError

def find_original_index(packed_index, lengths):
    total = 0
    for row, length in enumerate(lengths):
        if packed_index < total + length:
            return (row, packed_index - total)
        total += length
    raise IndexError("Packed index out of range")

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

import time 

class DiffusionSampler(Sampler):
    def __init__(self, method, model, noise, batch_dims, token_dim, BOS, strategy, strategy_para=None, eps=1e-5, device=torch.device('cuda'), schedule='uniform'):
        super().__init__(model, batch_dims, token_dim, strategy, strategy_para, device)
        self.noise = noise
        self.eps = eps
        self.method = method
        self.update_cnt = 0
        self.BOS = BOS
        self.token_dim = token_dim
        self.schedule = schedule 

    @torch.no_grad()
    def sample(self, steps, proj_fun=lambda x: x):
        if self.strategy == 'direct':
            return self.direct_sample(steps, proj_fun)
        elif self.strategy == 'top_p':
            return self.strateged_sample(steps, proj_fun)

    @torch.no_grad()
    def strateged_sample0(self, steps, proj_fun=lambda x: x):
        self.model.eval()

        batchsize, seq_len = self.batch_dims # seq_len: model length
        max_len = seq_len + 50 # for generated tokens exceeding the model length e.g. 1024

        # init x, t=1
        x = [[self.BOS] for _ in range(batchsize)]
        packed_x_tensor = torch.tensor(list(chain(*x)), dtype=torch.int32).to(self.device)
        seqlens = torch.tensor([len(xi) for xi in x] , dtype=torch.int32).to(self.device)

        # unpacked tis
        unpacked_tis = torch.zeros((batchsize, max_len, self.token_dim + 1), dtype=torch.float64).to(self.device) # (B, max_len, V + 1)

        timesteps = torch.linspace(1, self.eps, steps + 1, device=self.device)
        for i in range(steps):
            range_push(f'steps {i}')
            t = timesteps[i]
            sigma = self.get_sigma(t).repeat(batchsize)
            update_rate = self.get_update_rate(t, steps) if i < steps - 1 else 1 + 1e-3

            seqid = torch.repeat_interleave(torch.arange(batchsize).to(self.device), seqlens)

            # unpacked x tensor 
            unpack_mask = torch.arange(max_len, device=self.device)[None, :] < seqlens[:, None]  # (B, max_len)

            # NFE
            unpacked_changed_mask = unpack_mask #& changed_mask[:, None]  # (B, max_len)
            rows, cols = torch.where(unpacked_changed_mask)
            unpacked_tis[rows, cols, :-1] = self.model(packed_x_tensor, seqlens, sigma).double().exp() 
            
            packed_tis = unpacked_tis[unpack_mask]

            # probs
            probs = packed_tis * update_rate
            VOID = self.token_dim

            range_push('samp')
            # sampling insertions
            # insertions = sample_categorical(probs.to(torch.float64)).int()

            # location sampling
            insertions = torch.zeros_like(packed_x_tensor).int()

            prob_void = 1 - probs.sum(-1, keepdim=False)
            void_locations = torch.rand_like(prob_void) < prob_void
            insertions[void_locations] = VOID 

            probs[~void_locations] /= probs[~void_locations].sum(-1, keepdim=True)


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


            insertions[~void_locations] = top_p_sampling(probs[~void_locations], p=self.strategy_para).int()

            # print(f'{insertions.shape, prob_void.shape, probs[~void_locations].shape = }')

            range_pop() 

            range_push('ins')
            # update seqlens
            seqlens.scatter_add_(0, seqid, (insertions != VOID).int())

            # update x 
            inserted_with_void = torch.stack((packed_x_tensor, insertions), dim=1).view(-1) # (2 * packed, )
            packed_x_tensor = inserted_with_void[inserted_with_void != VOID]

            range_pop()
            range_pop()
            
        res = [_x.tolist() for _x in torch.split(packed_x_tensor, seqlens.tolist())]

        return [_x[1:] for _x in res] 
    
    @torch.no_grad()
    def strateged_sample(self, steps, proj_fun=lambda x: x):
        self.model.eval()

        batchsize, seq_len = self.batch_dims # seq_len: model length
        max_len = seq_len + 50 # for generated tokens exceeding the model length 

        # init x 
        x = [[self.BOS] for _ in range(batchsize)]
        packed_x_tensor = torch.tensor(list(chain(*x)), dtype=torch.int32).to(self.device)
        seqlens = torch.tensor([len(xi) for xi in x] , dtype=torch.int32).to(self.device)

        # cache
        changed_mask = torch.ones(batchsize, dtype=torch.bool).to(self.device)

        # unpacked tis
        unpacked_tis = torch.zeros((batchsize, max_len, self.token_dim + 1), dtype=torch.float64).to(self.device) # (B, max_len, V + 1)

        timesteps = torch.linspace(1, self.eps, steps + 1, device=self.device)
        for i in range(steps):
            t = timesteps[i]
            update_rate = self.get_update_rate(t, steps) if i < steps - 1 else 1 + 1e-3

            seqid = torch.repeat_interleave(torch.arange(batchsize).to(self.device), seqlens) # (packed, )
            unpack_mask = torch.arange(max_len, device=self.device)[None, :] < seqlens[:, None]  # (B, max_len)

            # score
            if changed_mask.any():
                packed_changed_mask = changed_mask[seqid] 
                packed_changed_x_tensor = packed_x_tensor[packed_changed_mask]
                changed_seqlens = seqlens[changed_mask]

                # NFE
                unpacked_changed_mask = unpack_mask & changed_mask[:, None]  # (B, max_len)
                rows, cols = torch.where(unpacked_changed_mask)
                unpacked_tis[rows, cols, :-1] = self.model(packed_changed_x_tensor, changed_seqlens).double().exp() 
                packed_tis = unpacked_tis[unpack_mask]

            # probs
            probs = packed_tis * update_rate
            # probs[..., -1]  = 1 - probs.sum(-1, keepdim=False)
            VOID = self.token_dim

            # sampling insertions
            # insertions = sample_categorical(probs.to(torch.float64)).int()#.tolist()
            # location sampling
            # insertions = torch.zeros_like(packed_x_tensor).int()
            insertions = torch.full_like(packed_x_tensor, fill_value=VOID, dtype=torch.int32)

            prob_void = 1 - probs.sum(-1, keepdim=False)
            void_locations = torch.rand_like(prob_void) < prob_void

            probs_non_void_locations = probs[~void_locations]
            probs_non_void_locations /= probs_non_void_locations.sum(-1, keepdim=True)

            insertions[~void_locations] = top_p_sampling(probs_non_void_locations, p=self.strategy_para).int()


            # update seqlens
            seqlens_old = seqlens.clone()
            seqlens.scatter_add_(0, seqid, (insertions != VOID).int())

            # update x 
            inserted_with_void = torch.stack((packed_x_tensor, insertions), dim=1).view(-1) # (2 * packed, )
            packed_x_tensor = inserted_with_void[inserted_with_void != VOID]

            changed_mask = seqlens != seqlens_old

        res = [_x.tolist() for _x in torch.split(packed_x_tensor, seqlens.tolist())]
        return [_x[1:-1] for _x in res] 


    @torch.no_grad()
    def direct_sample(self, steps, proj_fun=lambda x: x):
        self.model.eval()

        batchsize, seq_len = self.batch_dims # seq_len: model length
        max_len = seq_len + 50 # for generated tokens exceeding the model length 

        # init x 
        x = [[self.BOS] for _ in range(batchsize)]
        packed_x_tensor = torch.tensor(list(chain(*x)), dtype=torch.int32).to(self.device)
        seqlens = torch.tensor([len(xi) for xi in x] , dtype=torch.int32).to(self.device)

        # cache
        changed_mask = torch.ones(batchsize, dtype=torch.bool).to(self.device)

        # unpacked tis
        unpacked_tis = torch.zeros((batchsize, max_len, self.token_dim + 1), dtype=torch.float64).to(self.device) # (B, max_len, V + 1)

        timesteps = torch.linspace(1, self.eps, steps + 1, device=self.device)
        
        if self.schedule == 'cosine':
            timesteps = torch.cos(torch.pi / 2 * (1 - timesteps)) # dense -> sparse

        total_d1 = 0
        total_d2 = 0 

        for i in range(steps):
            t0 = time.perf_counter()

            t = timesteps[i]
            dt = timesteps[i] - timesteps[i+1] 
            update_rate = self.get_update_rate(t, steps, dt) if i < steps - 1 else 1 + 1e-3

            seqid = torch.repeat_interleave(torch.arange(batchsize).to(self.device), seqlens) # (packed, )
            unpack_mask = torch.arange(max_len, device=self.device)[None, :] < seqlens[:, None]  # (B, max_len)

            
            # score
            if changed_mask.any():
                packed_changed_mask = changed_mask[seqid] 
                packed_changed_x_tensor = packed_x_tensor[packed_changed_mask]
                changed_seqlens = seqlens[changed_mask]

                # NFE
                unpacked_changed_mask = unpack_mask & changed_mask[:, None]  # (B, max_len)
                rows, cols = torch.where(unpacked_changed_mask)
                unpacked_tis[rows, cols, :-1] = self.model(packed_changed_x_tensor, changed_seqlens).double().exp() 
                packed_tis = unpacked_tis[unpack_mask]

            
            t1 = time.perf_counter()

            # probs
            probs = packed_tis * update_rate
            probs[..., -1]  = 1 - probs.sum(-1, keepdim=False)
            VOID = self.token_dim

            # sampling insertions
            insertions = sample_categorical(probs.to(torch.float64)).int()#.tolist()

            # update seqlens
            seqlens_old = seqlens.clone()
            seqlens.scatter_add_(0, seqid, (insertions != VOID).int())

            # update x 
            inserted_with_void = torch.stack((packed_x_tensor, insertions), dim=1).view(-1) # (2 * packed, )
            packed_x_tensor = inserted_with_void[inserted_with_void != VOID]

            changed_mask = seqlens != seqlens_old

            t2 = time.perf_counter()

            total_d1 += t1 - t0 
            total_d2 += t2 - t1

        res = [_x.tolist() for _x in torch.split(packed_x_tensor, seqlens.tolist())]
        return [_x[1:-1] for _x in res] , total_d1, total_d2
    
    @torch.no_grad()
    def warmup(self, steps=1, proj_fun=lambda x: x):
        self.model.eval()

        batchsize, seq_len = self.batch_dims # seq_len: model length
        max_len = seq_len + 50 # for generated tokens exceeding the model length e.g. 1024

        # init x, t=1
        x = [[self.BOS] * max_len for _ in range(batchsize)]
        packed_x_tensor = torch.tensor(list(chain(*x)), dtype=torch.int32).to(self.device)
        seqlens = torch.tensor([len(xi) for xi in x] , dtype=torch.int32).to(self.device)

        # mem = -1 * torch.ones((batchsize * max_len,), dtype=torch.int32).to(self.device) # memory allocated for packed x tensor
        # mem[:seqlens.sum()] = torch.tensor(list(chain(*x)), dtype=torch.int32).to(self.device)

        # for cache
        changed_mask = torch.ones(batchsize, dtype=torch.bool).to(self.device)

        # unpacked tis
        unpacked_tis = torch.zeros((batchsize, max_len, self.token_dim + 1), dtype=torch.float64).to(self.device) # (B, max_len, V + 1)

        timesteps = torch.linspace(1, self.eps, steps + 1, device=self.device)
        for i in range(1):
            t = timesteps[i]
            update_rate = self.get_update_rate(t, steps) if i < steps - 1 else 1 + 1e-3

            seqid = torch.repeat_interleave(torch.arange(batchsize).to(self.device), seqlens)

            # unpacked x tensor 
            unpack_mask = torch.arange(max_len, device=self.device)[None, :] < seqlens[:, None]  # (B, max_len)
            # packed_x_tensor = mem[:seqlens.sum()]

            # score
            if changed_mask.any():
                packed_changed_mask = changed_mask[seqid] 

                # packed changed x tensor, changed seqlens 
                packed_changed_x_tensor = packed_x_tensor[packed_changed_mask]
                changed_seqlens = seqlens[changed_mask]

                # NFE
                unpacked_changed_mask = unpack_mask & changed_mask[:, None]  # (B, max_len)
                rows, cols = torch.where(unpacked_changed_mask)
                unpacked_tis[rows, cols, :-1] = self.model(packed_changed_x_tensor, changed_seqlens).double().exp() 
                
                packed_tis = unpacked_tis[unpack_mask]

            # probs
            probs = packed_tis * update_rate
            probs[..., -1]  = 1 - probs.sum(-1, keepdim=False)
            VOID = self.token_dim
            # probs = torch.cat((probs, probs_void), -1)

            range_push('samp')
            # sampling insertions
            insertions = sample_categorical(probs.to(torch.float64)).int()#.tolist()
            range_pop() 

            range_push('ins')
            # update seqlens
            seqlens_old = seqlens.clone()
            seqlens_after_insertion = seqlens.scatter_add(0, seqid, (insertions != VOID).int())

            # 
            insertions[(seqlens_after_insertion > seq_len)[seqid]] = VOID 
            seqlens.scatter_add_(0, seqid, (insertions != VOID).int())

            # update x 
            inserted_with_void = torch.stack((packed_x_tensor, insertions), dim=1).view(-1) # (2 * packed, )
            packed_x_tensor = inserted_with_void[inserted_with_void != VOID]

            changed_mask = seqlens != seqlens_old
            range_pop()

        res = [_x.tolist() for _x in torch.split(packed_x_tensor, seqlens.tolist())]

        return [_x[1:-1] for _x in res]

    def get_update_rate(self, t, steps, dt):
        # dt = (1 - self.eps) / steps
        curr_sigma, next_sigma = self.noise(t)[0], self.noise(t - dt)[0]
        d_curr_sigma = self.noise(t)[1]
        if self.method == 'tweedie':
            update_rate = ((-next_sigma).exp() - (-curr_sigma).exp()) / (1 - (-curr_sigma).exp())
        elif self.method == 'euler':
            update_rate = dt * d_curr_sigma * (-curr_sigma).exp() / (1 - (-curr_sigma).exp())
        return update_rate
    

class OrderedSampler(Sampler):
    def __init__(self, model, batch_dims, token_dim, strategy, strategy_para=None, order=None, device=torch.device('cuda')):
        super().__init__(model, batch_dims, token_dim, strategy, strategy_para, device)
        self.order = order

    @torch.no_grad()
    def sample(self, steps, proj_fun=lambda x: x):
        order = torch.randperm(1024) if self.order is None else self.order
        self.model.eval()
        x = (self.token_dim - 1) * torch.ones(*self.batch_dims, dtype=torch.int64).to(self.device)
        x = proj_fun(x)

        for i in range(steps):
            logits = self.model.logits(x)
            update_logits = logits[:, order[i], :-1]
            x[:, order[i]] = sample_with_strategy(update_logits, self.strategy, self.strategy_para)
        return x
