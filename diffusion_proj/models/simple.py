from diffusion_proj.util.storage import save, load
from diffusion_proj.util.fancy_text import cmod, Fore

from torch import Tensor
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def calc_padding(kernel_size: int, stride=1, dilation=1):
    return math.ceil(((kernel_size - 1) * dilation - (stride - 1)) / 2)


class ResBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int, kernel_size: int, time_embeddings: int, label_embeddings: int = None,
                 stride=2, downsample=True, padding_mode='circular', bias=False):
        super().__init__()
        assert stride <= 2

        # Build - Time
        self.time_embedding = nn.Embedding(time_embeddings, outputs)
        # Build - Labels
        self.label_embedding = nn.Embedding(label_embeddings, outputs) if label_embeddings is not None else None
        # Build - Convolution
        padding_in = calc_padding(kernel_size, stride)
        padding_out = calc_padding(kernel_size, 1)
        self.norm1 = nn.BatchNorm2d(inputs)
        if downsample:
            self.inp_proj = nn.Conv2d(inputs, outputs, kernel_size=kernel_size, stride=stride,
                                      padding=padding_in, padding_mode=padding_mode, bias=bias)
            self.res_proj = nn.Conv2d(inputs, outputs, kernel_size=1, stride=stride, bias=bias) \
                if outputs != inputs or stride > 1 else None
        else:
            self.inp_proj = nn.ConvTranspose2d(inputs, outputs, kernel_size=kernel_size, stride=stride,
                                               padding=padding_in, output_padding=stride // 2,
                                               padding_mode='zeros', bias=bias)
            self.res_proj = nn.ConvTranspose2d(inputs, outputs, kernel_size=1, stride=stride, padding=0,
                                               output_padding=stride // 2, padding_mode='zeros', bias=bias) \
                if outputs != inputs or stride > 1 else None
        self.mul_proj = nn.Conv2d(outputs, outputs, kernel_size=kernel_size, padding=padding_out,
                                  padding_mode=padding_mode, bias=bias)
        self.norm2 = nn.BatchNorm2d(outputs)
        self.out_proj = nn.Conv2d(outputs, outputs, kernel_size=kernel_size, padding=padding_out,
                                  padding_mode=padding_mode, bias=bias)
        self.actv = nn.SiLU()

        # Attributes
        self.input_dims = inputs
        self.output_dims = outputs
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.downsample = downsample
        self.upsample = not downsample
        self.labels = label_embeddings is not None

    def forward(self, image: Tensor, time_step: Tensor, **kwargs):
        verbose = kwargs.get('verbose')
        if verbose:
            print(f"Input Image = {image.shape}")
            if verbose >= 2:
                print(f"Input Timestep = {time_step.shape}")
        tensor_images = self.inp_proj(self.norm1(image))
        if verbose:
            print(f"Image Tensor = {tensor_images.shape}")
        tensor_time = self.time_embedding(time_step)
        if verbose:
            print(f"Time Embedding = {tensor_time.shape}")
        tensor_images = tensor_images + tensor_time.view(*tensor_time.shape, 1, 1)
        batch_labels = kwargs.get('labels')
        if self.labels and batch_labels is not None:
            if verbose and verbose >= 2:
                print(f"Labels Tensor = {batch_labels.shape}")
            tensor_labels = self.label_embedding(batch_labels)
            if verbose:
                print(f"Label Embedding = {tensor_labels.shape}")
            tensor_images = tensor_images + tensor_labels.view(*tensor_labels.shape, 1, 1)
        tensor_images = self.actv(self.mul_proj(self.norm2(self.actv(tensor_images))))
        if verbose:
            print(f"Image Tensor = {tensor_images.shape}")
        continuous_residual = image if self.res_proj is None else self.res_proj(image)
        level_residual = kwargs.get('residual')
        if self.upsample and level_residual is not None:
            if verbose and verbose >= 2:
                print(f"Level Residual = {level_residual.shape}")
            continuous_residual = level_residual if self.res_proj is None else self.res_proj(level_residual)
        if verbose:
            print(f"Residual = {continuous_residual.shape}")
        tensor_images = self.out_proj(tensor_images)
        if verbose:
            print(cmod(f"Output Image = {tensor_images.shape}", Fore.MAGENTA))
        tensor_images = tensor_images + continuous_residual
        return tensor_images

    def __repr__(self):
        return f"{self.__class__.__name__}({self.input_dims}, {self.output_dims}, kernel_size={self.kernel_size}, stride={self.stride}, " \
               f"bias={self.bias}, timesteps={self.time_embedding}, labels={self.label_embedding}, atcv={self.actv}, downsample={self.downsample})"


class UNet(nn.Module):
    def __init__(self, img_channels: int, kernel_size: int, sequence_channels: Union[int, list[int]],
                 time_embeddings: int, label_embeddings: int = None, layers: Union[int, list[int]] = 1,
                 kernel_size_init: Union[int, None] = 7, padding_mode='circular', bias=False,
                 device='cpu', dtype=torch.float32):
        super().__init__()
        if isinstance(sequence_channels, (int, float)):
            sequence_channels = [int(sequence_channels)]
        if isinstance(sequence_channels, tuple):
            sequence_channels = list(sequence_channels)
        if isinstance(layers, (int, float)):
            layers = [layers for _ in range(len(sequence_channels) - 1)]
        layers = layers[:len(sequence_channels)]
        if kernel_size_init is None:
            kernel_size_init = kernel_size

        self.encode = nn.Conv2d(img_channels, sequence_channels[0], kernel_size=kernel_size_init,
                                stride=1, padding=calc_padding(kernel_size_init, 1), bias=bias,
                                padding_mode=padding_mode)
        self.pre_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.down_sampling: list[ResBlock] = nn.ModuleList()
        for seq_idx, (channels_in, channels_out) in enumerate(zip(sequence_channels, sequence_channels[1:])):
            for layer_idx in range(layers[seq_idx]):
                self.down_sampling.append(
                    ResBlock(channels_in if layer_idx == 0 else channels_out, channels_out,
                             kernel_size, time_embeddings, label_embeddings, stride=2 if layer_idx == 0 else 1,
                             downsample=True, padding_mode=padding_mode, bias=bias)
                )
        self.up_sampling: list[ResBlock] = nn.ModuleList()
        sequence_reverse = sequence_channels[::-1]
        for seq_idx, (channels_in, channels_out) in enumerate(zip(sequence_reverse, sequence_reverse[1:])):
            for layer_idx in range(layers[seq_idx]):
                self.up_sampling.append(
                    ResBlock(channels_in, channels_out if layer_idx == layers[seq_idx] - 1 else channels_in,
                             kernel_size, time_embeddings, label_embeddings,
                             stride=2 if layer_idx == layers[seq_idx] - 1 else 1,
                             downsample=False, padding_mode=padding_mode, bias=bias)
                )
        self.norm = nn.BatchNorm2d(sequence_channels[0])
        # self.post_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.decode = nn.Conv2d(sequence_channels[0], img_channels, kernel_size, 1, calc_padding(kernel_size, 1),
                                padding_mode=padding_mode, bias=bias)
        self.to(device, dtype)
        self.eval()

        # Attributes
        self.img_channels = img_channels
        self.seq_channels: list[int] = list(sequence_channels)
        self.layers: list[int] = layers
        self.kernel_size = kernel_size
        self.kernel_size_init = kernel_size_init
        self.bias = bias
        self.time_embeddings = time_embeddings
        self.label_embeddings = label_embeddings

    def forward(self, images: Tensor, timesteps: Tensor, **kwargs):
        assert images.ndim >= 3 and timesteps.ndim == 1
        verbose = kwargs.get('verbose')
        level_residuals: list[Tensor] = []
        if verbose:
            print(cmod(f"Input Images = {images.shape}", Fore.CYAN))
        tensor = self.encode(images)
        tensor = F.silu(self.pre_pool(tensor) + tensor)
        if verbose:
            print(cmod(f"Encoded Images = {tensor.shape}", Fore.CYAN))
        for i, ds in enumerate(self.down_sampling):
            try:
                tensor = ds(tensor, timesteps, **kwargs)
            except RuntimeError as e:
                print(cmod(f"On Module {ds}, ie: Step {i}, last_shape={tensor.shape}", Fore.LIGHTYELLOW_EX))
                raise e
            level_residuals.append(tensor)
        if verbose:
            print(cmod(f"UNet Bottom Images = {tensor.shape}", Fore.CYAN))
        for i, (us, res) in enumerate(zip(self.up_sampling, reversed(level_residuals))):
            try:
                kwargs['residual'] = res  # if i != 0 else None
                tensor = us(tensor, timesteps, **kwargs)
            except RuntimeError as e:
                print(cmod(f"On Module {us}, ie: Step {i}, last_shape={tensor.shape}", Fore.LIGHTYELLOW_EX))
                raise e
        tensor = self.decode(self.norm(tensor))
        if verbose:
            print(cmod(f"Decoded Images = {tensor.shape}", Fore.CYAN))
        return tensor

    def save(self, file_no: int = None):

        name = f"params_globals-i{self.img_channels}-s{tuple([d for d in self.seq_channels])}-" \
               f"l{tuple([l for l in self.layers])}-k{self.kernel_size}-ki{self.kernel_size_init}-b{self.bias}-" \
               f"t{self.time_embeddings}-a{self.label_embeddings}"
        params = self.state_dict()
        save(params, name, 'mnist_diffusion_models', subdirectory=f"{self.__class__.__name__}", file_no=file_no)

    def load(self, file_no: int = None):
        name = f"params_globals-i{self.img_channels}-s{tuple([d for d in self.seq_channels])}-" \
               f"l{tuple([l for l in self.layers])}-k{self.kernel_size}-ki{self.kernel_size_init}-b{self.bias}-" \
               f"t{self.time_embeddings}-a{self.label_embeddings}"
        params = load(name, 'mnist_diffusion_models', subdirectory=f"{self.__class__.__name__}", file_no=file_no)
        if params is not None:
            self.load_state_dict(params)
