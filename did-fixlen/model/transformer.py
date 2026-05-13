import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import flash_attn
from einops import rearrange
from torch.nn.functional import scaled_dot_product_attention
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf

from . import rotary, rotary_scaled
from torch.cuda.nvtx import range_push, range_pop, mark 

#################################################################################
#                                  Layers                                       #
#################################################################################
class LayerNormWot(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.bias = nn.Parameter(torch.zeros([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=False):
            x = F.layer_norm(x.float(), self.weight.shape, self.weight, self.bias, 1e-5) # .float()
        return x 

#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlockWot(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4, dropout=0.1, use_checkpoint=False):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNormWot(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNormWot(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True), 
            nn.GELU(approximate="tanh"), 
            nn.Linear(mlp_ratio * dim, dim, bias=True)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout

        self.use_checkpoint = use_checkpoint

    def forward(self, x, seqlens,cu_seqlens,  rotary_cos_sin):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, seqlens, cu_seqlens, rotary_cos_sin)
        else:
            return self._forward(x, seqlens, cu_seqlens, rotary_cos_sin)

    def _forward(self, x, seqlens, cu_seqlens,  rotary_cos_sin):
        x_skip = x
        x = self.norm1(x)

        # qkv
        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 't (three h d) ->  t three h d', three=3, h=self.n_heads)
        q = qkv[:, 0]
        k = qkv[:, 1]
        v = qkv[:, 2]
        
        # qk = qkv[:, :2]
        # qk = rearrange(qk, 't two h d ->  t (two h) d', two=2, h=self.n_heads)
# 
        # rotary
        with torch.amp.autocast("cuda", enabled=False):
            cos, sin = rotary_cos_sin 
            max_seqlen = cos.shape[0]
            q = flash_attn.layers.rotary.apply_rotary_emb(q, cos.to(qkv.dtype), sin.to(qkv.dtype), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            k = flash_attn.layers.rotary.apply_rotary_emb(k, cos.to(qkv.dtype), sin.to(qkv.dtype), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            # q = flash_attn.layers.rotary.apply_rotary_emb(q, cos, sin, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            # k = flash_attn.layers.rotary.apply_rotary_emb(k, cos, sin, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            # qk = flash_attn.layers.rotary.apply_rotary_emb(qk, cos.to(qkv.dtype), sin.to(qkv.dtype), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

        # q, k = rearrange(qk, 't (two h) d ->  two t h d', two=2, h=self.n_heads)

        # attention
        x = flash_attn.flash_attn_interface.flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen, 
            causal=False
        )
        x = rearrange(x, 't h d -> t (h d)')
        
        # out
        x = x_skip.to(x.dtype) + F.dropout(self.attn_out(x), p=self.dropout, training=self.training)

        # mlp
        x = torch.add(x, F.dropout(self.mlp(self.norm2(x)), p=self.dropout, training=self.training))
        return x
        

class DDiTBlockWotScaledRoPE(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4, dropout=0.1, use_checkpoint=False):
        super().__init__()
        self.n_heads = n_heads

        self.norm1 = LayerNormWot(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNormWot(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True), 
            nn.GELU(approximate="tanh"), 
            nn.Linear(mlp_ratio * dim, dim, bias=True)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout

        self.use_checkpoint = use_checkpoint

    def forward(self, x, seqlens, rotary_cos_sin):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, seqlens, rotary_cos_sin)
        else:
            return self._forward(x, seqlens, rotary_cos_sin)

    def _forward(self, x, seqlens, rotary_cos_sin):
        x_skip = x
        x = self.norm1(x)

        # qkv
        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 't (three h d) -> t three h d', three=3, h=self.n_heads)
        
        # rotary
        with torch.amp.autocast("cuda", enabled=False):
            cos, sin = rotary_cos_sin 
            qkv = rotary_scaled.apply_rotary_pos_emb(qkv, cos, sin)
        qkv = qkv.to(torch.float16)
        q, k, v = rearrange(qkv, 't three h d -> three t h d', three=3, h=self.n_heads)
        
        # attention
        cu_seqlens = seqlens.cumsum(-1).to(torch.int32)
        cu_seqlens = torch.cat([
            torch.zeros(1, dtype=cu_seqlens.dtype, device=cu_seqlens.device),  
            cu_seqlens
        ]) 
        max_seqlen = seqlens.max()
        
        x = flash_attn.flash_attn_interface.flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen, 
            causal=False
        )
            
        x = rearrange(x, 't h d -> t (h d)')
        
        # out
        x = x_skip.to(x.dtype) + F.dropout(self.attn_out(x), p=self.dropout, training=self.training)

        # mlp
        x = torch.add(x, F.dropout(self.mlp(self.norm2(x)), p=self.dropout, training=self.training))
        return x
     

class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        """
        Mode arg: 0 -> use a learned layer, 1 -> use eigenvectors,
        2-> add in eigenvectors, 3 -> use pretrained embedding matrix
        """
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        return self.embedding[x]


class DDitFinalLayerWotLin(nn.Module):
    def __init__(self, hidden_size, out_channels, clamp_max):
        super().__init__()
        self.norm_final = LayerNormWot(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)

        self.logV = math.log(out_channels)
        self.clamp_max = clamp_max
        
    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        x = (x - self.logV).clamp(max=self.clamp_max)
        return x

class DDitFinalLayerWotCsoftmax(nn.Module):
    def __init__(self, hidden_size, out_channels, clamp_max):
        super().__init__()
        self.norm_final = LayerNormWot(hidden_size)

        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear2 = nn.Linear(hidden_size, 1)

        self.clamp_max = clamp_max
        
    def forward(self, x):
        x = self.norm_final(x)
        log_N_norm = self.linear(x).log_softmax(-1)
        log_L = self.linear2(x).clamp(max=self.clamp_max)
        return log_N_norm + log_L

class _DDitFinalLayerWotSeqNorm(nn.Module):
    """
    sequence-level normalization: the summation of all probabilities at all positions should be |x_0| - |x_t|
    """
    def __init__(self, hidden_size, out_channels, max_len):
        super().__init__()
        self.norm_final = LayerNormWot(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.max_len = max_len

    def forward(self, x, seqlens):
        """
        x: (packed_tokens, hidden_dim)
        seqlens: (batch_size,)
        """
        x = self.norm_final(x)

        # log N norm
        log_N_norm = self.linear(x).log_softmax(-1)

        # log L norm
        log_L = self.linear2(x)
        B = len(seqlens)
        mask = torch.arange(self.max_len + 50, device=x.device)[None, :] < seqlens[:, None]  # (B, max_len)
        unpacked = torch.full((B, self.max_len + 50, 1), -torch.inf, device=log_L.device, dtype=log_L.dtype) # (B, max_len, 1)
        unpacked[mask] = log_L 
        log_L_norm = F.log_softmax(unpacked, dim=1) # max_len dim

        # log(|x_0| - |x_t|)
        adjustments = torch.log((self.max_len - seqlens).clamp(min=0) + 1e-4).view(B, 1, 1)  # (B, 1, 1)

        return log_N_norm + (log_L_norm + adjustments)[mask]


# with prompt
class DDitFinalLayerWotSeqNorm(nn.Module):
    """
    sequence-level normalization: the summation of all probabilities at all positions should be |x_0| - |x_t|
    """
    def __init__(self, hidden_size, out_channels, max_len):
        super().__init__()
        self.norm_final = LayerNormWot(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.max_len = max_len

    def forward(self, x, seqlens, seqlens_init=None):
        """
        x: (packed_tokens, hidden_dim)
        seqlens: (batch_size,)
        """
        x = self.norm_final(x)

        # log N norm
        log_N_norm = self.linear(x).log_softmax(-1)

        # log L norm
        log_L = self.linear2(x)
        B = len(seqlens)
        mask = torch.arange(self.max_len + 50, device=x.device)[None, :] < seqlens[:, None]  # (B, max_len)
        unpacked = torch.full((B, self.max_len + 50, 1), -torch.inf, device=log_L.device, dtype=log_L.dtype) # (B, max_len, 1)
        unpacked[mask] = log_L 
        
        if seqlens_init is not None:
            prompt_mask = torch.arange(self.max_len + 50, device=x.device)[None, :] >= seqlens_init[:, None] - 1  # (B, max_len)
            unpacked[~prompt_mask] = -torch.inf
        
        log_L_norm = F.log_softmax(unpacked, dim=1) # max_len dim

        # log(|x_0| - |x_t|)
        adjustments = torch.log((self.max_len - seqlens).clamp(min=0) + 1e-4).view(B, 1, 1)  # (B, 1, 1)

        return log_N_norm + (log_L_norm + adjustments)[mask]


class DID(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()

        # hack to make loading in configs easier
        if type(config) == dict:
            config = OmegaConf.create(config)

        self.config = config

        vocab_size = config.tokens + config.dummy_class

        self.vocab_embed = EmbeddingLayer(config.model.hidden_size, vocab_size)
        self.seq_len = config.model.length
        self.scaled_rope = config.scaled_rope
        if config.scaled_rope:
            self.rotary_emb = rotary_scaled.Rotary(config.model.hidden_size // config.model.n_heads, self.seq_len)
            self.blocks = nn.ModuleList(
                [
                    DDiTBlockWotScaledRoPE(
                        config.model.hidden_size, config.model.n_heads, dropout=config.model.dropout, use_checkpoint=config.model.use_checkpoint
                    )
                    for _ in range(config.model.n_blocks)
                ]
            )
        else:
            self.rotary_emb = rotary.Rotary(config.model.hidden_size // config.model.n_heads)
            self.blocks = nn.ModuleList(
                [
                    DDiTBlockWot(
                        config.model.hidden_size, config.model.n_heads, dropout=config.model.dropout, use_checkpoint=config.model.use_checkpoint
                    )
                    for _ in range(config.model.n_blocks)
                ]
            )

        if config.final_layer == 'csoftmax':
            self.output_layer = DDitFinalLayerWotCsoftmax(config.model.hidden_size, vocab_size, clamp_max=math.log(self.seq_len))
        elif config.final_layer == 'linear':
            self.output_layer = DDitFinalLayerWotLin(config.model.hidden_size, vocab_size, clamp_max=math.log(self.seq_len))
        elif config.final_layer == 'seqnorm':
            self.output_layer = DDitFinalLayerWotSeqNorm(config.model.hidden_size, vocab_size, max_len=self.seq_len)

        # tie weight
        if config.tie_weight:
            self.vocab_embed.embedding = self.output_layer.linear.weight

        if config.model.dtype == 'float32':
            self.dtype = torch.float32
        elif config.model.dtype == 'float16':
            self.dtype = torch.float16
        elif config.model.dtype == 'bfloat16':
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.bfloat16
    
    def forward(self, packed_tokens, seqlens, seqlens_init=None):
        # packed_tokens: (t, ), seqlens: (b, )

        x = self.vocab_embed(packed_tokens) # (t, h*d)
        
        # rope
        if self.scaled_rope:
            cos, sin = self.rotary_emb(seqlens) # 2 * (p, 3, 1, d) -> qkv: (p, 3, h, d) 
        else:
            cos, sin = self.rotary_emb(x.device, self.seq_len + 100) # 2 * (1, s, 3, 1, d) -> qkv: (b, s, 3, h, d) 
            cos = cos[0,:,0,0,:cos.shape[-1]//2] 
            sin = sin[0,:,0,0,:sin.shape[-1]//2]
        rotary_cos_sin = cos, sin

        # cu_seqlens
        cu_seqlens = seqlens.cumsum(-1).to(torch.int32)
        cu_seqlens = torch.cat([
            torch.zeros(1, dtype=cu_seqlens.dtype, device=cu_seqlens.device),  
            cu_seqlens
        ]) 

        with torch.amp.autocast("cuda", dtype=self.dtype):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, seqlens, cu_seqlens, rotary_cos_sin)

            if isinstance(self.output_layer, DDitFinalLayerWotSeqNorm):
                x = self.output_layer(x, seqlens, seqlens_init)
            else:
                x = self.output_layer(x)

        if self.config.dummy_class:
            return x[..., :-1]
        else:
            return x