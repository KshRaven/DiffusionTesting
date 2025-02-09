
from diffusion_proj.model.base.sub import VAE_AttentionBlock, VAE_ResidualBlock
from diffusion_proj.model.base.debugging import get_tensor_info
from diffusion_proj.util.fancy_text import CM, Fore

from torch import nn
from torch.nn import functional as F
from torch import Tensor

import torch


class VAE_Encoder(nn.Module):
    def __init__(self, inp_channels: int = 3, out_channels: int = 4, embed_size=128, heads=1,
                 extra_depth: int = None, max_mult_steps: int = None, max_stride_steps: int = None,
                 pad_reductions: bool = True, norm_groups: int = 32,
                 bias=True, device='cpu', dtype=torch.float32):
        super(VAE_Encoder, self).__init__()
        out_channels *= 2
        steps: list[nn.Module] = [
            nn.Conv2d(inp_channels, embed_size, 3, padding=1, bias=bias, device=device, dtype=dtype),
            VAE_ResidualBlock(embed_size, embed_size, norm_groups, bias, device, dtype),
            VAE_ResidualBlock(embed_size, embed_size, norm_groups, bias, device, dtype),
        ]
        if extra_depth is None:
            extra_depth = 0
        if max_mult_steps is None:
            max_mult_steps = extra_depth
        if max_stride_steps is None:
            max_stride_steps = extra_depth
        embed_size_new = embed_size
        for i in range(extra_depth):
            if i < max_mult_steps:
                embed_size_new *= 2
            if i < max_stride_steps:
                stride = 2
            else:
                stride = 1
            steps.append(nn.Conv2d(embed_size, embed_size, 3, stride, 1 if pad_reductions else 0, bias=bias, device=device, dtype=dtype))
            steps.append(VAE_ResidualBlock(embed_size, embed_size_new, norm_groups, bias, device, dtype))
            steps.append(VAE_ResidualBlock(embed_size_new, embed_size_new, norm_groups, bias, device, dtype))
            embed_size = embed_size_new
        steps.append(VAE_ResidualBlock(embed_size_new, embed_size_new, norm_groups, bias, device, dtype))
        steps.append(VAE_AttentionBlock(embed_size_new, heads, norm_groups, bias, device, dtype))
        steps.append(VAE_ResidualBlock(embed_size_new, embed_size_new, norm_groups, bias, device, dtype))
        steps.append(nn.GroupNorm(norm_groups, embed_size_new, device=device, dtype=dtype))
        steps.append(nn.SiLU())
        steps.append(nn.Conv2d(embed_size_new, out_channels, kernel_size=3, padding=1, bias=bias, device=device, dtype=dtype))
        steps.append(nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, bias=bias, device=device, dtype=dtype))

        self.steps           = nn.ModuleList(steps)
        self.constant        = nn.Parameter(torch.tensor(1.0, device=device, dtype=dtype))
        self.inp_features    = inp_channels
        self.out_features    = out_channels / 2
        self.size_reductions = max_stride_steps
        self.pad_reductions  = pad_reductions

    def forward(self, tensor: Tensor, noise: Tensor, debug=False):
        # tensor: (Batch_Size, Channel, Height, Width)
        # noise: (Batch_Size, 4, Height / 8, Width / 8)

        # Apply transformations -> (Batch_Size, 8, Height / 8, Width / 8)
        # debugged_att = False
        for i, module in enumerate(self.steps):
            try:
                if not self.pad_reductions or (isinstance(module, nn.Conv2d) and module.stride == (2, 2)):
                    padding = (0, 1 if tensor.shape[-2] % 2 != 0 else 0, 0, 1 if tensor.shape[-1] % 2 != 0 else 0)
                    tensor = F.pad(tensor, padding, 'circular')
                tensor = module(tensor)
                # if debug and not debugged_att and isinstance(module, VAE_AttentionBlock):
                #     module(tensor, debug=True)
                #     debugged_att = True
            except RuntimeError as e:
                print(CM(f"On Module {module}, ie: Step {i+1}", Fore.LIGHTYELLOW_EX))
                raise e
        if debug:
            print(get_tensor_info(tensor, "VAE Encoder ~ Transformation"))

        # (Batch_Size, 8, Height / 8, Width / 8) -> two tensors of shape (Batch_Size, 4, Height / 8, Width / 8)
        mean, log_variance = torch.chunk(tensor, 2, dim=1)
        # Clamp the log variance between -30 and 20, so that the variance is between (circa) 1e-14 and 1e8. 
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        log_variance = torch.clamp(log_variance, -30, 20)
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        variance = log_variance.exp()
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        stdev = variance.sqrt()
        if debug:
            print(get_tensor_info(mean, "VAE Encoder ~ Mean"))
            print(get_tensor_info(stdev, "VAE Encoder ~ Std Dev"))
            print(get_tensor_info(noise, "VAE Encoder ~ Noise"))
        
        # Transform N(0, 1) -> N(mean, stdev) 
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        tensor = mean + stdev * noise
        if debug:
            print(get_tensor_info(tensor, "VAE Encoder ~ Noisy Tensor"))

        # Scale by a constant
        # Constant taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L17C1-L17C1
        # tensor *= 0.18215
        tensor *= self.constant
        if debug:
            print(get_tensor_info(self.constant, "VAE Encoder ~ Constant"))
        
        return tensor


if __name__ == '__main__':
    test_enc = VAE_Encoder(extra_depth=3, max_mult_steps=2, max_stride_steps=None)
    test_enc.eval()

    with torch.no_grad():
        print(test_enc)
        test_tensor = torch.rand(1, 3, 28, 28)
        test_res = test_enc(test_tensor, torch.randn(1, 4, 4, 4), debug=True)
        print(get_tensor_info(test_res, "Output"))
