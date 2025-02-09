
from build.model.base.sub import SelfAttention, CrossAttention

from torch import nn
from torch.nn import functional as F
from torch import Tensor

import torch


class TimeEmbedding(nn.Module):
    def __init__(self, embed_size: int, fwd_exp: int = 4, bias=True, device='cpu', dtype=torch.float32):
        super().__init__()
        self.linear_1 = nn.Linear(embed_size, fwd_exp * embed_size, bias, device, dtype)
        self.linear_2 = nn.Linear(fwd_exp * embed_size, fwd_exp * embed_size, bias, device, dtype)

        self.embed_size = embed_size
        self.fwd_exp = fwd_exp

    def forward(self, x):
        # x: (1, 320)

        # (1, 320) -> (1, 1280)
        x = self.linear_1(x)
        
        # (1, 1280) -> (1, 1280)
        x = F.silu(x) 
        
        # (1, 1280) -> (1, 1280)
        x = self.linear_2(x)

        return x


class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_time=1280, norm_groups: int = 32,
                 bias=True, device='cpu', dtype=torch.float32):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(norm_groups, in_channels, device=device, dtype=dtype)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=bias, device=device, dtype=dtype)
        self.linear_time = nn.Linear(n_time, out_channels, bias, device, dtype)

        self.groupnorm_merged = nn.GroupNorm(norm_groups, out_channels, device=device, dtype=dtype)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=bias, device=device, dtype=dtype)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=bias, device=device, dtype=dtype)
    
    def forward(self, feature: Tensor, time: Tensor):
        # feature: (Batch_Size, In_Channels, Height, Width)
        # time: (1, 1280)

        residue = feature
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature = self.groupnorm_feature(feature)
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature = F.silu(feature)
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        feature = self.conv_feature(feature)
        
        # (1, 1280) -> (1, 1280)
        time = F.silu(time)

        # (1, 1280) -> (1, Out_Channels)
        time = self.linear_time(time)
        
        # Add width and height dimension to time. 
        # (Batch_Size, Out_Channels, Height, Width) + (1, Out_Channels, 1, 1) -> (Batch_Size, Out_Channels, Height, Width)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = self.groupnorm_merged(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = F.silu(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = self.conv_merged(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) + (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        return merged + self.residual_layer(residue)


class UNET_AttentionBlock(nn.Module):
    def __init__(self, embed_size: int, heads: int, context_size: int = 768, norm_groups: int = 32, fwd_exp: int = 4,
                 bias=True, device='cpu', dtype=torch.float32):
        super().__init__()
        channels = heads * embed_size
        
        self.groupnorm = nn.GroupNorm(norm_groups, channels, eps=1e-6, device=device, dtype=dtype)
        self.conv_input = nn.Conv2d(channels, channels, 1, padding=0, bias=bias, device=device, dtype=dtype)

        self.layernorm_1 = nn.LayerNorm(channels, bias=bias, device=device, dtype=dtype)
        self.attention_1 = SelfAttention(channels, heads, bias, device, dtype)
        self.layernorm_2 = nn.LayerNorm(channels, bias=bias, device=device, dtype=dtype)
        self.attention_2 = CrossAttention(channels, context_size, heads, bias, device, dtype)
        self.layernorm_3 = nn.LayerNorm(channels, bias=bias, device=device, dtype=dtype)
        self.linear_geglu_1 = nn.Linear(channels, fwd_exp * channels * 2, bias, device, dtype)
        self.linear_geglu_2 = nn.Linear(fwd_exp * channels, channels, bias, device, dtype)

        self.conv_output = nn.Conv2d(channels, channels, 1, padding=0, bias=bias, device=device, dtype=dtype)
    
    def forward(self, tensor: Tensor, context: Tensor):
        # x: (Batch_Size, Features, Height, Width)
        # context: (Batch_Size, Seq_Len, Dim)

        residue_long = tensor

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        tensor = self.groupnorm(tensor)
        
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        tensor = self.conv_input(tensor)
        
        n, c, h, w = tensor.shape
        
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        tensor = tensor.view((n, c, h * w))
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features)
        tensor = tensor.transpose(-1, -2)
        
        # Normalization + Self-Attention with skip connection

        # (Batch_Size, Height * Width, Features)
        residue_short = tensor
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        tensor = self.layernorm_1(tensor)
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        tensor = self.attention_1(tensor)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        tensor += residue_short
        
        # (Batch_Size, Height * Width, Features)
        residue_short = tensor

        # Normalization + Cross-Attention with skip connection
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        tensor = self.layernorm_2(tensor)
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        tensor = self.attention_2(tensor, context)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        tensor += residue_short
        
        # (Batch_Size, Height * Width, Features)
        residue_short = tensor

        # Normalization + FFN with GeGLU and skip connection
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        tensor = self.layernorm_3(tensor)
        
        # GeGLU as implemented in the original code: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/attention.py#L37C10-L37C10
        # (Batch_Size, Height * Width, Features) -> two tensors of shape (Batch_Size, Height * Width, Features * 4)
        tensor, gate = self.linear_geglu_1(tensor).chunk(2, dim=-1)
        
        # Element-wise product: (Batch_Size, Height * Width, Features * 4) * (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features * 4)
        tensor = tensor * F.gelu(gate)
        
        # (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features)
        tensor = self.linear_geglu_2(tensor)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        tensor += residue_short
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        tensor = tensor.transpose(-1, -2)
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        tensor = tensor.view((n, c, h, w))

        # Final skip connection between initial input and output of the block
        # (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        return self.conv_output(tensor) + residue_long


class Upsample(nn.Module):
    def __init__(self, channels, bias=True, device='cpu', dtype=torch.float32):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=bias, device=device, dtype=dtype)
    
    def forward(self, tensor: Tensor):
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * 2, Width * 2)
        tensor = F.interpolate(tensor, scale_factor=2, mode='nearest')
        return self.conv(tensor)


class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


class UNET(nn.Module):
    def __init__(self, inp_channels: int = 4, out_channels: int = 3, embed_size=128, heads=1, n_time=1280,
                 extra_depth: int = None, max_mult_steps: int = None, max_stride_steps: int = None,
                 pad_reductions: bool = True, context_size: int = 768, norm_groups: int = 32, fwd_exp=4, embed_div=8,
                 bias=True, device='cpu', dtype=torch.float32):
        super().__init__()
        self.encoders = nn.ModuleList([
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 16, Width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            
            # (Batch_Size, 320, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 32, Width / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),
            
            # (Batch_Size, 640, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])

        embed_size_ = embed_size

        if extra_depth is None:
            extra_depth = 0
        if max_mult_steps is None:
            max_mult_steps = extra_depth
        if max_stride_steps is None:
            max_stride_steps = extra_depth

        enc = [
            SwitchSequential(nn.Conv2d(inp_channels, embed_size, 3, 1, 1, bias=bias, device=device, dtype=dtype)),
        ]

        embed_size_new = embed_size
        for i in range(extra_depth):
            print(embed_size, embed_size_new)
            if extra_depth - (i + 1) < max_mult_steps:
                embed_size_new *= 2
            if i < max_stride_steps:
                stride = 2
            else:
                stride = 1
            enc.append(SwitchSequential(
                UNET_ResidualBlock(embed_size, embed_size_new, n_time, norm_groups, bias, device, dtype),
                UNET_AttentionBlock(embed_size_new // embed_div, heads, context_size, norm_groups, fwd_exp, bias, device, dtype)
            ))
            embed_size = embed_size_new
            enc.append(SwitchSequential(
                UNET_ResidualBlock(embed_size, embed_size, n_time, norm_groups, bias, device, dtype),
                UNET_AttentionBlock(embed_size // embed_div, heads, context_size, norm_groups, fwd_exp, bias, device, dtype)
            ))
            enc.append(SwitchSequential(
                nn.Conv2d(embed_size, embed_size, 3, stride, 1, bias=bias, device=device, dtype=dtype)
            ))
        enc.append(SwitchSequential(
            UNET_ResidualBlock(embed_size, embed_size, n_time, norm_groups, bias, device, dtype),
        ))
        enc.append(SwitchSequential(
            UNET_ResidualBlock(embed_size, embed_size, n_time, norm_groups, bias, device, dtype),
        ))

        self.bottleneck = SwitchSequential(
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_ResidualBlock(embed_size, embed_size),
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_AttentionBlock(8, 160), 
            
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_ResidualBlock(1280, 1280), 
        )
        
        self.decoders = nn.ModuleList([
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 32, Width / 32) 
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
            
            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1920, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),
            
            # (Batch_Size, 1920, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 1280, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 960, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),
            
            # (Batch_Size, 960, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])

    def forward(self, x, context, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim) 
        # time: (1, 1280)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)
        
        return x


class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # x: (Batch_Size, 320, Height / 8, Width / 8)

        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = self.groupnorm(x)
        
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = F.silu(x)
        
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = self.conv(x)
        
        # (Batch_Size, 4, Height / 8, Width / 8) 
        return x


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)
    
    def forward(self, latent, context, time):
        # latent: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim)
        # time: (1, 320)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)
        
        # (Batch, 4, Height / 8, Width / 8) -> (Batch, 320, Height / 8, Width / 8)
        output = self.unet(latent, context, time)
        
        # (Batch, 320, Height / 8, Width / 8) -> (Batch, 4, Height / 8, Width / 8)
        output = self.final(output)
        
        # (Batch, 4, Height / 8, Width / 8)
        return output
