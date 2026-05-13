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



# Flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

# function overload
def modulate(x: torch.Tensor,
             shift: torch.Tensor,
             scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift

@torch.jit.script
def modulate_fused(x: torch.Tensor,
                   shift: torch.Tensor,
                   scale: torch.Tensor) -> torch.Tensor:
    return modulate(x, shift, scale)


import typing
def bias_dropout_add_scale(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float,
    training: bool) -> torch.Tensor:
  if bias is not None:
    out = scale * F.dropout(x + bias, p=prob, training=training)
  else:
    out = scale * F.dropout(x, p=prob, training=training)

  if residual is not None:
    out = residual + out
  return out


def get_bias_dropout_add_scale(training):
  def _bias_dropout_add(x, bias, scale, residual, prob):
    return bias_dropout_add_scale(
      x, bias, scale, residual, prob, training)

  return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_scale_fused_train(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, True)


@torch.jit.script
def bias_dropout_add_scale_fused_inference(
    x: torch.Tensor,
    bias: typing.Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: typing.Optional[torch.Tensor],
    prob: float) -> torch.Tensor:
  return bias_dropout_add_scale(
    x, bias, scale, residual, prob, False)

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

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=False):
            x = F.layer_norm(x.float(), self.weight.shape, self.weight)
        return x 



#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=3.5, dropout=0.1, use_checkpoint=False, cond_dim=128):
        super().__init__()
        self.n_heads = n_heads

        # self.norm1 = LayerNormWot(dim)
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        # self.norm2 = LayerNormWot(dim)
        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(mlp_ratio * dim), bias=True), 
            nn.GELU(approximate="tanh"), 
            nn.Linear(int(mlp_ratio * dim), dim, bias=True)
        )
        self.dropout2 = nn.Dropout(dropout)
        self.dropout = dropout

        self.use_checkpoint = use_checkpoint

        
        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()


    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(self, x, seqlens,cu_seqlens,  rotary_cos_sin, c):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, seqlens, cu_seqlens, rotary_cos_sin, c)
        else:
            return self._forward(x, seqlens, cu_seqlens, rotary_cos_sin, c)

    def _forward(self, x, seqlens, cu_seqlens,  rotary_cos_sin, c):
        
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        (shift_msa, scale_msa, gate_msa, shift_mlp,
        scale_mlp, gate_mlp) = torch.repeat_interleave(self.adaLN_modulation(c), seqlens, dim=0).chunk(6, dim=-1)

        # print(f'{shift_msa.shape, scale_msa.shape = }'); exit(0)

        x_skip = x
        # x = self.norm1(x)
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)

        # qkv
        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 't (three h d) ->  t three h d', three=3, h=self.n_heads)
        q = qkv[:, 0]
        k = qkv[:, 1]
        v = qkv[:, 2]

        # rotary
        with torch.amp.autocast("cuda", enabled=False):
            cos, sin = rotary_cos_sin 
            max_seqlen = cos.shape[0]
            q = flash_attn.layers.rotary.apply_rotary_emb(q, cos.to(qkv.dtype), sin.to(qkv.dtype), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            k = flash_attn.layers.rotary.apply_rotary_emb(k, cos.to(qkv.dtype), sin.to(qkv.dtype), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

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
        
        # # out
        # x = x_skip.to(x.dtype) + F.dropout(self.attn_out(x), p=self.dropout, training=self.training)
        x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)

        # # mlp
        # x = torch.add(x, F.dropout(self.mlp(self.norm2(x)), p=self.dropout, training=self.training))
        x = bias_dropout_scale_fn(self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)), None, gate_mlp, x, self.dropout)

        return x
        


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True)
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            - math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

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


class DDitFinalLayerCsoftmax(nn.Module):
    def __init__(self, hidden_size, out_channels, clamp_max, cond_dim=128):
        super().__init__()
        # self.norm_final = LayerNormWot(hidden_size)
        self.norm_final = LayerNorm(hidden_size)

        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear2 = nn.Linear(hidden_size, 1)

        self.clamp_max = clamp_max

        # from mdlm
        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()
        
    def forward(self, x, seqlens, c):
        # x = self.norm_final(x)
        
        shift, scale = torch.repeat_interleave(self.adaLN_modulation(c), seqlens, dim=0).chunk(2, dim=-1)
        
        
        x = modulate_fused(self.norm_final(x), shift, scale)


        log_N_norm = self.linear(x).log_softmax(-1)
        log_L = self.linear2(x).clamp(max=self.clamp_max)
        return log_N_norm + log_L

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
        
        self.sigma_map = TimestepEmbedder(config.model.cond_dim)
        
        self.rotary_emb = rotary.Rotary(config.model.hidden_size // config.model.n_heads)
        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    config.model.hidden_size, config.model.n_heads, dropout=config.model.dropout, use_checkpoint=config.model.use_checkpoint, cond_dim=config.model.cond_dim
                )
                for _ in range(config.model.n_blocks)
            ]
        )

        if config.final_layer == 'csoftmax':
            self.output_layer = DDitFinalLayerCsoftmax(config.model.hidden_size, vocab_size, clamp_max=math.log(self.seq_len), cond_dim=config.model.cond_dim)

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
    
    def forward(self, packed_tokens, seqlens, t):
        # packed_tokens: (p, ), seqlens: (b, ), t: (b,) 

        x = self.vocab_embed(packed_tokens) # (p, h*d)

        
        c = F.silu(self.sigma_map(t))
        
        # rope
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
                range_push(f'{i} layer')
                x = self.blocks[i](x, seqlens, cu_seqlens, rotary_cos_sin, c)
                range_pop()

            range_push(f'final layer')
            x = self.output_layer(x, seqlens, c)
            range_pop()

        if self.config.dummy_class:
            return x[..., :-1]
        else:
            return x