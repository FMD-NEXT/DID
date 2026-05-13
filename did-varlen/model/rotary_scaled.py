import torch
from torch import nn


class Rotary(torch.nn.Module):
    def __init__(self, dim, max_len=1024, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_len = max_len

    def forward(self, seqlens):
        device = seqlens.device 

        indices = torch.arange(self.max_len, device=device)[None, :] # (1, S)
        mask = indices < seqlens[:, None]  # (B, S)
        packed_indices = indices.expand_as(mask)[mask] # (packed, )
        packed_total_len = torch.repeat_interleave(seqlens, seqlens) # (packed, )

        scaled_indices = packed_indices / packed_total_len * self.max_len # (packed, )

        freqs = torch.einsum("i,j->ij", scaled_indices, self.inv_freq.clone()) # (packed, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # dims are: packed, qkv, head, dim
        cos = emb.cos()[:, None, None, :].repeat(1,3,1,1)
        sin = emb.sin()[:, None, None, :].repeat(1,3,1,1)

        # This makes the transformation on v an identity.
        cos[:,2,:,:].fill_(1.)
        sin[:,2,:,:].fill_(0.)

        return cos, sin

# @torch.jit.script
def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat(
        (-x2, x1), dim=-1
    )

@torch.jit.script
def _apply_rotary_pos_emb_torchscript(qkv, cos, sin):
    return (qkv * cos) + (rotate_half(qkv) * sin)


# @torch.jit.script
def apply_rotary_pos_emb(qkv, cos, sin):
    return _apply_rotary_pos_emb_torchscript(qkv, cos, sin)
