
from build.model.base.debugging import get_tensor_info

from torch import nn
from torch.nn import functional as F
from torch import Tensor

import torch
import numpy as np


class SelfAttention(nn.Module):
    def __init__(self, embed_size: int, heads: int, bias=True, device='cpu', dtype=torch.float32):
        super().__init__()
        # This combines the Wq, Wk and Wv matrices into one matrix
        self.in_proj    = nn.Linear(embed_size, 3 * embed_size, bias=bias, device=device, dtype=dtype)
        # This one represents the Wo matrix
        self.out_proj   = nn.Linear(embed_size, embed_size, bias=bias, device=device, dtype=dtype)
        self.head_num   = heads
        self.embed_size = embed_size // heads

    def forward(self, tensor: Tensor, causal_mask: bool = False, debug: bool = False):
        # tensor: (Batch_Size, Seq_Len, Dim)

        # (Batch_Size, Seq_Len, Dim)
        input_shape = tensor.shape

        # (Batch_Size, Seq_Len, Dim)
        batch_size, sequence_length, d_embed = input_shape

        # (Batch_Size, Seq_Len, H, Dim / H)
        interim_shape = (batch_size, sequence_length, self.head_num, self.embed_size)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3) -> 3 tensor of shape (Batch_Size, Seq_Len, Dim)
        q, k, v = self.in_proj(tensor).chunk(3, dim=-1)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)
        if debug:
            print(get_tensor_info(q, "Self Attention ~ Q View"))
            print(get_tensor_info(v, "Self Attention ~ V View"))

        # (Batch_Size, H, Seq_Len, Dim / H) @ (Batch_Size, H, Dim / H, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = q @ k.transpose(-1, -2)
        if debug:
            print(get_tensor_info(weight, "Self Attention ~ Energy"))

        if causal_mask:
            # Mask where the upper triangle (above the principal diagonal) is 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            # Fill the upper triangle with -inf
            weight.masked_fill_(mask, -torch.inf)
            if debug:
                print(get_tensor_info(weight, "Self Attention ~ Energy Mask Fill"))

        # Divide by d_k (Dim / H).
        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight /= np.sqrt(self.embed_size)

        # (Batch_Size, H, Seq_Len, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = F.softmax(weight, dim=-1)
        if debug:
            print(get_tensor_info(weight, "Self Attention ~ Attention"))

        # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        output = weight @ v
        if debug:
            print(get_tensor_info(output, "Self Attention ~ Attented Values"))

        # (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, Seq_Len, H, Dim / H)
        output = output.transpose(1, 2)

        # (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, Seq_Len, Dim)
        output = output.reshape(input_shape)

        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim)
        output = self.out_proj(output)
        if debug:
            print(get_tensor_info(output, "Self Attention ~ Output"))

        # (Batch_Size, Seq_Len, Dim)
        return output


class CrossAttention(nn.Module):
    def __init__(self, embed_size: int, cross_embed_size: int, heads: int,
                 bias=True, device='cpu', dtype=torch.float32):
        super().__init__()
        self.q_proj     = nn.Linear(embed_size, embed_size, bias=bias, device=device, dtype=dtype)
        self.k_proj     = nn.Linear(cross_embed_size, embed_size, bias=bias, device=device, dtype=dtype)
        self.v_proj     = nn.Linear(cross_embed_size, embed_size, bias=bias, device=device, dtype=dtype)
        self.out_proj   = nn.Linear(embed_size, embed_size, bias=bias, device=device, dtype=dtype)
        self.head_num   = heads
        self.embed_size = embed_size // heads

    def forward(self, latent: Tensor, context: Tensor, debug: bool = False):
        # latent (latent): (Batch_Size, Seq_Len_Q, Dim_Q)
        # context (context): (Batch_Size, Seq_Len_KV, Dim_KV) = (Batch_Size, 77, 768)

        input_shape = latent.shape
        batch_size, sequence_length, d_embed = input_shape
        # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
        interim_shape = (batch_size, -1, self.head_num, self.embed_size)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        q = self.q_proj(latent)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        k = self.k_proj(context)
        # (Batch_Size, Seq_Len_KV, Dim_KV) -> (Batch_Size, Seq_Len_KV, Dim_Q)
        v = self.v_proj(context)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        q = q.view(interim_shape).transpose(1, 2)
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        k = k.view(interim_shape).transpose(1, 2)
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        v = v.view(interim_shape).transpose(1, 2)
        if debug:
            print(get_tensor_info(q, "Cross Attention ~ Latent View"))
            print(get_tensor_info(v, "Cross Attention ~ Context VIew"))

        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) @ (Batch_Size, H, Dim_Q / H, Seq_Len_KV) -> (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = q @ k.transpose(-1, -2)
        if debug:
            print(get_tensor_info(weight, "Cross Attention ~ Energy"))

        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight /= np.sqrt(self.embed_size)

        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV)
        weight = F.softmax(weight, dim=-1)
        if debug:
            print(get_tensor_info(weight, "Cross Attention ~ Attention"))

        # (Batch_Size, H, Seq_Len_Q, Seq_Len_KV) @ (Batch_Size, H, Seq_Len_KV, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        output = weight @ v
        if debug:
            print(get_tensor_info(output, "Cross Attention ~ Attented Values"))

        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H)
        output = output.transpose(1, 2).contiguous()

        # (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = output.view(input_shape)

        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        output = self.out_proj(output)
        if debug:
            print(get_tensor_info(output, "Cross Attention ~ Output"))

        # (Batch_Size, Seq_Len_Q, Dim_Q)
        return output


class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int, heads: int, norm_groups: int = 32, bias=True, device='cpu', dtype=torch.float32):
        super().__init__()
        self.groupnorm = nn.GroupNorm(norm_groups, channels, device=device, dtype=dtype)
        self.attention = SelfAttention(channels, heads, bias, device, dtype)

    def forward(self, tensor: Tensor, debug=False):
        # x: (Batch_Size, Features, Height, Width)

        residue = tensor
        if debug:
            print(get_tensor_info(tensor, "Self Attention ~ Input"))

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        tensor = self.groupnorm(tensor)

        n, c, h, w = tensor.shape

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        tensor = tensor.view((n, c, h * w))

        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features). Each pixel becomes a feature of size "Features", the sequence length is "Height * Width".
        tensor = tensor.transpose(-1, -2)

        # Perform self-attention WITHOUT mask
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        tensor = self.attention(tensor, debug=debug)

        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        tensor = tensor.transpose(-1, -2)

        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        tensor = tensor.view((n, c, h, w))

        # (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        tensor += residue
        if debug:
            print(get_tensor_info(tensor, "Self Attention ~ Output"))

        # (Batch_Size, Features, Height, Width)
        return tensor


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_groups: int = 32,
                 bias=True, device='cpu', dtype=torch.float32):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(norm_groups, in_channels, device=device, dtype=dtype)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=bias, device=device, dtype=dtype)

        self.groupnorm_2 = nn.GroupNorm(norm_groups, out_channels, device=device, dtype=dtype)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=bias, device=device, dtype=dtype)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=bias, device=device, dtype=dtype)

        self.norm_groups = norm_groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias

    def forward(self, tensor: Tensor):
        # x: (Batch_Size, In_Channels, Height, Width)

        residue = tensor

        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        tensor = self.groupnorm_1(tensor)

        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        tensor = F.silu(tensor)

        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        tensor = self.conv_1(tensor)

        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        tensor = self.groupnorm_2(tensor)

        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        tensor = F.silu(tensor)

        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        tensor = self.conv_2(tensor)

        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        return tensor + self.residual_layer(residue)

    # def extra_repr(self) -> str:
    #     return f'in_channels={self.in_channels}, out_channels={self.out_channels}, ' \
    #            f'norm_groups={self.norm_groups}, bias={self.bias}'
