
from diffusion_proj.util.fancy_text import cmod, Fore

from torch import Tensor

import torch

COLOURS = {lbl: clr for lbl, clr in vars(Fore).items() if
           not any(fltr in lbl for fltr in ['BLACK', 'WHITE']) and 'LIGHT' in lbl}


class color_fetch:
    def __init__(self):
        self.current_idx = 0
        self.colors = list(COLOURS.values())
        self.color_num = len(self.colors)

    def __call__(self):
        self.current_idx += 1
        return self.colors[self.current_idx % self.color_num]


get = color_fetch()


def get_tensor_info(tensor: Tensor, label: str = None, show_tensor=True, dim=None):
    tensor = tensor.detach().clone().cpu().type(torch.float32)
    skip = '\n'
    return f"{f'{cmod(label, get())} =>{skip}' if label is not None else ''}" \
           f"{f'{tensor}{skip}' if show_tensor else ''}" \
           f"\tshape={tensor.shape}, mean={tensor.mean(dim)}, std={tensor.std(dim)}"


if __name__ == '__main__':
    print(COLOURS.items())

    for i in range(len(COLOURS)):
        print(get_tensor_info(torch.randn(3, 3), f"Tensor {i+1}"))
