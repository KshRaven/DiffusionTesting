
from torch import Tensor
from numpy import ndarray as CPUArray
from typing import Union

import torch
import numpy as np


class MemoryBuffer(object):
    def __init__(self, buffers: list[str] = None):
        if buffers is None:
            buffers = {}
        self.data: dict[str, list[Tensor]] = {b: list() for b in buffers}

    @property
    def episodes(self):
        if 'ep_map' in self.data:
            episode_mapping = self.data['ep_map']
            return len(np.unique(episode_mapping))
        else:
            return None

    @property
    def steps(self):
        return max([len(buffer) for buffer in self.data.values()])

    def add_buffers(self, *buffers: str):
        if len(buffers) == 1 and isinstance(buffers[0], (list, tuple)):
            buffers = buffers[0]
        for buffer in buffers:
            if buffer not in self.data:
                self.data[buffer] = list()
        self.reset()

    def update(self, validate=False, add_dims: list[int] = None, **buffers: Tensor):
        if validate:
            if any(b not in self.data for b in buffers.keys()):
                raise ValueError(f"Update all buffers at once")
        if isinstance(add_dims, (int, float)):
            add_dims = [int(add_dims)]
        for b, buffer in buffers.items():
            if isinstance(buffer, Tensor):
                buffer = buffer.detach().cpu()
                if add_dims is not None:
                    for dim in add_dims:
                        if isinstance(buffer, Tensor):
                            buffer = buffer.unsqueeze(dim)
                        elif isinstance(buffer, CPUArray):
                            buffer = np.expand_dims(buffer, dim)
            self.data[b].append(buffer)

    def reset(self):
        for b in self.data.keys():
            self.data[b] = list()

    def deque(self, to_del: list[int]):
        for name, buffer in self.data.items():
            self.data[name] = [item for idx, item in enumerate(buffer) if idx not in to_del]

    def __len__(self):
        buffers = list(self.data.keys())
        if len(buffers) == 0:
            return 0
        return max([len(self.data[buffer]) for buffer in buffers])

    def rollout(self, buffers: Union[str, list[str]] = None, sequence_length: int = None, as_list=False, stack=False,
                device: torch.device = None):
        if len(self.data) == 0:
            raise ValueError(f"No buffers created in replay.")
        if buffers is None:
            buffers = list(self.data.keys())
        elif isinstance(buffers, str):
            buffers = [buffers]
        for buffer in buffers:
            if buffer not in self.data:
                raise ValueError(f"Buffer '{buffer}' has not been added to replay.")
        if all([len(self.data[buffer]) == 0 for buffer in buffers]):
            raise ValueError(f"No data in replay buffers.")
        res = {}
        for label in buffers:
            buffer = self.data[label]
            if sequence_length is not None:
                buffer = buffer[-min(len(buffer), sequence_length):]
            if stack:
                if isinstance(buffer[0], CPUArray):
                    buffer = np.stack(buffer, axis=0)
                elif isinstance(buffer[0], Tensor):
                    buffer = torch.stack(buffer, dim=0)
                elif isinstance(buffer[0], (int, float, bool)):
                    pass
            if device is not None and isinstance(buffer, Tensor):
                buffer = buffer.to(device)
            res[label] = buffer
        if as_list:
            res = list(res.values())
        return res
