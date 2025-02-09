
from diffusion_proj.model.base.sub import VAE_AttentionBlock, VAE_ResidualBlock
from diffusion_proj.model.base.debugging import get_tensor_info
from diffusion_proj.util.fancy_text import CM, Fore

from torch import nn
from torch import Tensor

import torch


class VAE_Decoder(nn.Module):
    def __init__(self, inp_channels: int = 4, out_channels: int = 3, embed_size=128, heads=1,
                 extra_depth: int = None, max_mult_steps: int = None, max_stride_steps: int = None,
                 pad_reductions: bool = True, norm_groups: int = 32,
                 bias=True, device='cpu', dtype=torch.float32):
        super(VAE_Decoder, self).__init__()
        # out_channels *= 2
        if extra_depth is None:
            extra_depth = 0
        if max_mult_steps is None:
            max_mult_steps = extra_depth
        if max_stride_steps is None:
            max_stride_steps = extra_depth
        embed_size = embed_size * (2 ** max_mult_steps)
        steps: list[nn.Module] = [
            nn.Conv2d(inp_channels, inp_channels, 1, padding=0, bias=bias, device=device, dtype=dtype),
            nn.Conv2d(inp_channels, embed_size, 3, padding=1, bias=bias, device=device, dtype=dtype),
            VAE_ResidualBlock(embed_size, embed_size, norm_groups, bias, device, dtype),
            VAE_AttentionBlock(embed_size, heads, norm_groups, bias, device, dtype),
            VAE_ResidualBlock(embed_size, embed_size, norm_groups, bias, device, dtype),
        ]
        embed_size_new = embed_size
        for i in reversed(range(extra_depth)):
            print(embed_size, embed_size_new)
            steps.append(VAE_ResidualBlock(embed_size, embed_size_new, norm_groups, bias, device, dtype))
            steps.append(VAE_ResidualBlock(embed_size_new, embed_size_new, norm_groups, bias, device, dtype))
            steps.append(VAE_ResidualBlock(embed_size_new, embed_size_new, norm_groups, bias, device, dtype))
            if i < max_stride_steps:
                steps.append(nn.Upsample(scale_factor=2))
            steps.append(nn.Conv2d(embed_size_new, embed_size_new, 3, padding=1 if pad_reductions else 1, bias=bias, device=device, dtype=dtype))
            embed_size = embed_size_new
            if i < max_mult_steps:
                embed_size_new = embed_size // 2

        steps.append(VAE_ResidualBlock(embed_size, embed_size_new, norm_groups, bias, device, dtype))
        embed_size = embed_size_new
        steps.append(VAE_ResidualBlock(embed_size, embed_size, norm_groups, bias, device, dtype))
        steps.append(VAE_ResidualBlock(embed_size, embed_size, norm_groups, bias, device, dtype))
        steps.append(nn.GroupNorm(norm_groups, embed_size, device=device, dtype=dtype))
        steps.append(nn.SiLU())
        steps.append(nn.Conv2d(embed_size, out_channels, kernel_size=3, padding=1, bias=bias, device=device, dtype=dtype))

        self.steps = nn.ModuleList(steps)
        self.constant = nn.Parameter(torch.tensor(1.0, device=device, dtype=dtype))
        self.inp_features = inp_channels
        self.out_features = out_channels / 2
        self.size_reductions = max_stride_steps
        self.pad_reductions = pad_reductions

    def forward(self, tensor: Tensor, verbose: int = None):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        
        # Remove the scaling added by the Encoder.
        tensor /= self.constant
        if verbose:
            print(get_tensor_info(self.constant, "VAE Decoder Constant"))

        # debugged_att = False
        for i, module in enumerate(self.steps):
            try:
                tensor = module(tensor)
                # if debug and not debugged_att and isinstance(module, VAE_AttentionBlock):
                #     module(tensor, debug=True)
                #     debugged_att = True
            except RuntimeError as e:
                print(CM(f"On Module {module}, ie: Step {i}, last_shape={tensor.shape}", Fore.LIGHTYELLOW_EX))
                raise e
        if verbose:
            print(get_tensor_info(tensor, "VAE Decoder Transformation"))

        # (Batch_Size, 3, Height, Width)
        return tensor


if __name__ == '__main__':
    test_enc = VAE_Decoder(extra_depth=3, max_mult_steps=2, max_stride_steps=None)
    test_enc.eval()

    with torch.no_grad():
        print(test_enc)
        test_tensor = torch.rand(1, 4, 4, 4)
        test_res = test_enc(test_tensor, debug=True)
        print(get_tensor_info(test_res, "Output"))
