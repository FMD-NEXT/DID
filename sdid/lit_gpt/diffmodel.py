"""Full definition of a GPT NeoX Language Model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT and
https://github.com/EleutherAI/gpt-neox/tree/main/megatron/model.
"""
import math
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F 
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import Self
from flash_attn import flash_attn_func
from flash_attn.layers.rotary import apply_rotary_emb
from lit_gpt.config import Config
# from .fused_rotary_embedding import apply_rotary_emb_func
apply_rotary_emb_func = apply_rotary_emb
RoPECache = Tuple[torch.Tensor, torch.Tensor]
KVCache = Tuple[torch.Tensor, torch.Tensor]
FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")

### begin did 
import flash_attn

# adaln
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

# rotary
class Rotary(torch.nn.Module):
    def __init__(self, config, base=10_000):
        super().__init__()
        self.n_rope_elem = int(config.rotary_percentage * config.head_size)
        self.condense_ratio = config.condense_ratio if hasattr(config, 'condense_ratio') else 1
        self.seq_len = config.block_size #
        inv_freq = 1.0 / (base ** (torch.arange(0, self.n_rope_elem, 2).float() / self.n_rope_elem))
        self.register_buffer("inv_freq", inv_freq)

        seq_idx = torch.arange(self.seq_len) / self.condense_ratio
        idx_theta = torch.outer(seq_idx, self.inv_freq)
        cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x, seq_len=-1):
        seq_len = self.seq_len if seq_len == -1 else seq_len
        cos_cached = self.cos[:seq_len].to(x.device)
        sin_cached = self.sin[:seq_len].to(x.device)
        if x.dtype == torch.bfloat16:
            return cos_cached.bfloat16(), sin_cached.bfloat16()
        if x.dtype in (torch.float16, torch.bfloat16, torch.int8):
            return cos_cached.half(), sin_cached.half()
        return cos_cached, sin_cached

# attention
from einops import rearrange
class SelfAttention_VarLen(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.config = config
        self.scale_value = 1.0 / math.sqrt(config.head_size)

    def forward(self, x, rope, seqlens, cu_seqlens) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B = len(seqlens)
        S = seqlens.max().item()
        T, C = x.size()  # tokens, embedding dimensionality (n_embd)

        # qkv
        qkv = self.attn(x)
        q, k, v = qkv.split([self.config.n_head * self.config.head_size, self.config.n_query_groups * self.config.head_size, self.config.n_query_groups * self.config.head_size], dim=-1)
        q = q.reshape(T, -1, self.config.head_size)  # (T, nh_q, hs)
        k = k.reshape(T, -1, self.config.head_size)
        v = v.reshape(T, -1, self.config.head_size)
        
        # rotary
        with torch.amp.autocast("cuda", enabled=False):
            cos, sin = rope 
            max_seqlen = cos.shape[0]
            q = apply_rotary_emb(q, cos.to(qkv.dtype), sin.to(qkv.dtype), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
            k = apply_rotary_emb(k, cos.to(qkv.dtype), sin.to(qkv.dtype), cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

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
        return self.proj(x) 

# block with time
class Block_Wt(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = SelfAttention_VarLen(config)
        if not config.shared_attention_norm:
            self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config)
        self.config = config

        # adaln
        self.adaLN_modulation = nn.Linear(config.n_cond, 6 * config.n_embd, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()
        self.dropout = 0

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def forward(
        self,
        x,
        rope,
        c,
        seqlens, 
        cu_seqlens
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        bias_dropout_scale_fn = self._get_bias_dropout_scale()
        
        (shift_msa, scale_msa, gate_msa, shift_mlp,
        scale_mlp, gate_mlp) = torch.repeat_interleave(self.adaLN_modulation(c), seqlens, dim=0).chunk(6, dim=-1)

        # msa
        x = bias_dropout_scale_fn(self.attn(modulate_fused(self.norm_1(x), shift_msa, scale_msa), rope, seqlens, cu_seqlens), None, gate_msa, x, self.dropout)

        # mlp
        x = bias_dropout_scale_fn(self.mlp(modulate_fused(self.norm_2(x), shift_mlp, scale_mlp)), None, gate_mlp, x, self.dropout)

        return x

# time emb
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


# final layer
class DDitFinalLayerCsoftmax(nn.Module):
    def __init__(self, config, clamp_max):
        super().__init__()

        self.norm_final = config.norm_class(config.n_embd, eps=config.norm_eps)

        self.linear = nn.Linear(config.n_embd, config.padded_vocab_size)
        self.linear2 = nn.Linear(config.n_embd, 1)
        self.clamp_max = clamp_max

        # adaln
        self.adaLN_modulation = nn.Linear(config.n_cond, 2 * config.n_embd, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()
        
    def forward(self, x, c, seqlens):
        shift, scale = torch.repeat_interleave(self.adaLN_modulation(c), seqlens, dim=0).chunk(2, dim=-1)
        x = modulate_fused(self.norm_final(x), shift, scale)

        log_N_norm = self.linear(x).log_softmax(-1)
        log_L = self.linear2(x).clamp(max=self.clamp_max)
        return log_N_norm + log_L


# tfm encoder with time
class TransEncoder_Wt(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.sigma_map = TimestepEmbedder(config.n_cond)
        self.rotary = Rotary(config)   
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size + 1, config.n_embd),
                h=nn.ModuleList(Block_Wt(config) for _ in range(config.n_layer))
            )
        )     
        import math 
        self.lm_head = DDitFinalLayerCsoftmax(config, clamp_max=math.log(config.block_size))

        # self = torch.compile(self, dynamic=True) # meaningless

    def _init_weights(self, module: nn.Module, n_layer) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        # GPT-NeoX  https://arxiv.org/pdf/2204.06745.pdf
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
            # RWKV: set it to 1e-4
            # torch.nn.init.uniform_(module.weight,  -1e-4, 1e-4)
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # GPT-NeoX
        for name, p in module.named_parameters():
            if (name == "proj.weight" and isinstance(module, LLaMAMLP)) or (name == "w3.weight" and isinstance(module, SwiGLU) or (name=="proj.weight" and isinstance(module, SelfAttention))):  #if use xformer swiglu, fc2 layer will be renamed to w3
                nn.init.normal_(p, mean=0.0, std=1 / math.sqrt(self.config.n_embd)  /  n_layer)

    def forward(
        self, 
        idx: torch.Tensor, # (packed, )
        c, # (b,)
        seqlens  # (b,)
    ) -> torch.Tensor:
        # cu_seqlens
        cu_seqlens = seqlens.cumsum(-1).to(torch.int32)
        cu_seqlens = torch.cat([
            torch.zeros(1, dtype=cu_seqlens.dtype, device=cu_seqlens.device),  
            cu_seqlens
        ]) 
        
        # rope
        S = seqlens.max() 
        cos, sin = self.rotary(idx, seq_len=S)
        
        # cond
        c = F.silu(self.sigma_map(c))

        # tfm
        x = self.transformer.wte(idx)  # (p, n_embd)
        for block in self.transformer.h:
            x = block(x, (cos, sin), c, seqlens, cu_seqlens)
            
        return self.lm_head(x, c, seqlens)  # (B, S, V): logC + logsoftmax

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))



### end did

class TransEncoder(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        assert config.padded_vocab_size is not None
        self.config = config

        self.lm_head = nn.Linear(config.n_embd, config.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.padded_vocab_size + 1, config.n_embd),
                h=nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f=config.norm_class(config.n_embd, eps=config.norm_eps),
            )
        )
        self.rope_cache: Optional[RoPECache] = None

    def _init_weights(self, module: nn.Module, n_layer) -> None:
        """Meant to be used with `gpt.apply(gpt._init_weights)`."""
        # GPT-NeoX  https://arxiv.org/pdf/2204.06745.pdf
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
            # RWKV: set it to 1e-4
            # torch.nn.init.uniform_(module.weight,  -1e-4, 1e-4)
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / self.config.n_embd))
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        # GPT-NeoX
        for name, p in module.named_parameters():
            if (name == "proj.weight" and isinstance(module, LLaMAMLP)) or (name == "w3.weight" and isinstance(module, SwiGLU) or (name=="proj.weight" and isinstance(module, SelfAttention))):  #if use xformer swiglu, fc2 layer will be renamed to w3
                nn.init.normal_(p, mean=0.0, std=1 / math.sqrt(self.config.n_embd)  /  n_layer)


    def forward(
        self, idx: torch.Tensor
    ) -> torch.Tensor:
        B, T = idx.size()

        block_size = self.config.block_size
        assert block_size >= T, f"Cannot forward sequence of length {T}, block size is only {block_size}"

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(idx)
        # passing `attn_mask` to SDPA downgrades it to use the inefficient implementation. since we only need the mask
        # for the kv-cache support (only during inference), we only create it in that situation
        # this will be resolved by https://github.com/pytorch/pytorch/issues/96099

        cos, sin = self.rope_cache
        cos = cos[:T]
        sin = sin[:T]

        # forward the model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        for block in self.transformer.h:
            x = block(x, (cos, sin))

        x = self.transformer.ln_f(x)

        return self.lm_head(x)  # (b, t, vocab_size)

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        return cls(Config.from_name(name, **kwargs))

    def build_rope_cache(self, idx: torch.Tensor) -> RoPECache:
        return build_rope_cache(
            seq_len=self.config.block_size,
            n_elem=int(self.config.rotary_percentage * self.config.head_size),
            dtype=torch.bfloat16,
            device=idx.device,
            condense_ratio=self.config.condense_ratio,
        )


class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.attn = SelfAttention(config)
        if not config.shared_attention_norm:
            self.norm_2 = config.norm_class(config.n_embd, eps=config.norm_eps)
        self.mlp = config.mlp_class(config)
        self.config = config
    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:

        n_1 = self.norm_1(x)
        h = self.attn(n_1, rope)
        if self.config.parallel_residual:
            n_2 = n_1 if self.config.shared_attention_norm else self.norm_2(x)
            x = x + h + self.mlp(n_2)
        else:
            if self.config.shared_attention_norm:
                raise NotImplementedError(
                    "No checkpoint amongst the ones we support uses this configuration"
                    " (non-parallel residual and shared attention norm)."
                )

            x = x + h
            x = x + self.mlp(self.norm_2(x))
        return x


class SelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        # key, query, value projections for all heads, but in a batch
        self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        qkv = self.attn(x)

        # assemble into a number of query groups to support MHA, MQA and GQA together (see `config.n_query_groups`)
        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # each group has 1+ queries, 1 key, and 1 value
        qkv = qkv.view(B, T, self.config.n_query_groups, total_qkv, self.config.head_size) # (B, T, n_query_groups, total_qkv, hs)
        # qkv = qkv.permute(0, 2, 3, 1, 4)  # (B, n_query_groups, total_qkv, T, hs)

        # split batched computation into three
        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)

        # repeat k and v if necessary
        # Peiyuan: we do not need to do this as flash attention 2 already support GQA
        # if self.config.n_query_groups != 1:  # doing this would require a full kv cache with MQA (inefficient!)
        #     # for MHA this is a no-op
        #     k = k.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)
        #     v = v.expand(B, self.config.n_query_groups, q_per_kv, T, self.config.head_size)

        q = q.reshape(B,  T, -1, self.config.head_size)  # (B, T, nh_q, hs)
        k = k.reshape(B,  T, -1, self.config.head_size)
        v = v.reshape(B,  T, -1, self.config.head_size)

        cos, sin = rope

        # apply rope in fp32 significanly stabalize training
        # fused rope expect (batch_size, seqlen, nheads, headdim)
        q = apply_rotary_emb_func(q, cos, sin, False, True)
        k = apply_rotary_emb_func(k, cos, sin, False, True)

        # n_elem = int(self.config.rotary_percentage * self.config.head_size)

        # q_roped = apply_rope(q[..., :n_elem], cos.repeat(1,2), sin.repeat(1,2))
        # k_roped = apply_rope(k[..., :n_elem], cos.repeat(1,2), sin.repeat(1,2))
        # print( (q_roped - q).sum())
        # q = torch.cat((q_roped, q[..., n_elem:]), dim=-1)
        # k = torch.cat((k_roped, k[..., n_elem:]), dim=-1)

        y = self.scaled_dot_product_attention(q, k, v)

        y = y.reshape(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)

        return y

    def scaled_dot_product_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ):
        scale = 1.0 / math.sqrt(self.config.head_size)

        if (
            FlashAttention2Available
            and q.device.type == "cuda"
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            from flash_attn import flash_attn_func

            return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=False)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if q.size() != k.size():
             k = k.repeat_interleave(q.shape[1]//k.shape[1], dim=1)
             v = v.repeat_interleave(q.shape[1]//v.shape[1], dim=1)
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, scale=scale, is_causal=False
        )
        return y.transpose(1, 2)


class GptNeoxMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.gelu(x)
        return self.proj(x)


# swiglu
class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.fc_2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.silu(self.fc_1(x)) * self.fc_2(x)
        return self.proj(x)


# from xformers.ops import SwiGLU
# class LLaMAMLP(nn.Module):
#     def __init__(self, config: Config) -> None:
#         super().__init__()
#         # self.fc_1 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
#         # self.fc_2 = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
#         # self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)
#         self.swiglu = SwiGLU(config.n_embd,config.intermediate_size, bias=False, _pack_weights=False)
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x_fc_1 = self.fc_1(x)
#         # x_fc_2 = self.fc_2(x)
#         # x = torch.nn.functional.silu(x_fc_1) * x_fc_2
#         # return self.proj(x)
#         return self.swiglu(x)


def build_rope_cache(
    seq_len: int, n_elem: int, dtype: torch.dtype, device: torch.device, base: int = 10000, condense_ratio: int = 1
) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device) / n_elem))

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta)

    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)

    # added by peiyuan to ensure same data type with q, k, to use fused rotary embedding
    if dtype == torch.bfloat16:
        return cos.bfloat16(), sin.bfloat16()
    # this is to mimic the behaviour of complex32, else we will get different results
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        return cos.half(), sin.half()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.type_as(x)
