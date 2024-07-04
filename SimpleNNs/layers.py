"""
This file defines layers that are useful in
"""

import torch
from torch import Tensor


class Module:
    def __init__(self, device):
        self.device = device

    def forward(self, x: Tensor) -> Tensor:
        pass

    def parameters(self) -> list[Tensor]:
        params = []
        for _, param in self.__dict__.items():
            if isinstance(param, Tensor) and param.requires_grad:
                params.append(param)
            elif isinstance(param, Module):
                params.extend(param.parameters())
        return params

    def to(self, device) -> None:
        self.device = device
        for _, param in self.__dict__.items():
            if isinstance(param, Tensor) or isinstance(param, Module):
                param.to(device)


class Linear(Module):
    w: Tensor
    b: Tensor

    def __init__(self, in_dim, out_dim, device=torch.device('cpu')):
        super().__init__(device)
        self.w = torch.rand(in_dim, out_dim, requires_grad=True)
        self.b = torch.rand(out_dim, requires_grad=True)
        self.to(device)

    def forward(self, x):
        return torch.matmul(x, self.w) + self.b

    def parameters(self):
        return [self.w, self.b]


class Conv2d(Module):
    def __init__(self, in_channels:int, out_channels:int, kernel_size=3, stride=1,
                 padding=1, device=torch.device('cpu')):
        super().__init__(device)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.kernels = (torch.rand(out_channels, in_channels, kernel_size, kernel_size, requires_grad=True)
                       / (kernel_size * kernel_size))
        self.stride = stride
        self.padding = padding
        self.to(device)

    def forward(self, x: Tensor) -> Tensor:

        # add padding to x
        padded = torch.pad(x, (self.padding, self.padding, self.padding, self.padding))

        # build the windowed view to perform multiplication
        # this will build a tensor of shape [batch, in_channels, height, width, kernel size, kernel size]
        # so that each [i,j,k,l, :, :] index would be a kernel_size * kernel_size window to perform
        # multiplication with kernel
        windows = padded.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        # prepares for einsum
        # reshape into [batch, in_channel, height * width, kernel_size * kernel_size]
        windows = windows.contiguous().view(x.shape[0], x.shape[1], -1, self.kernel_size * self.kernel_size)

        # reshape into [out_channels, in_channels * kernel_size * kernel_size]
        kernel_view = self.kernels.view(self.out_channels, -1)

        output = torch.einsum("bijc,cj->bi", windows, kernel_view)

        # reshape to desired shape [batch_size, out_channels, out_height, out_width]
        out_height = (x.shape[3] + 2 * self.padding) // self.stride + 1
        out_width = (x.shape[4] + 2 * self.padding) // self.stride + 1
        output = output.view(x.shape[0], self.out_channels, out_height, out_width)
        return output


class MaxPool2d(Module):
    def __init__(self, height, width, device=torch.device('cpu')):
        super().__init__(device)
        self.height = height
        self.width = width

    def forward(self, x: Tensor) -> Tensor:
        # x is in shape [batch, channel, height, width]

        # make the view to be maximized, in shape [batch, channel, 
        # height/self.height, width/self.width, self.height, self.width]
        max_windows = x.unfold(2, self.height, self.height).unfold(3, self.width, self.width)

        # calculate the max and return
        return max_windows.amax(dim=(4, 5), keepdim=False)


class ReLU(Module):
    def forward(self, x:Tensor) -> Tensor:
        return torch.maximum(x, 0)


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return 1 / (1 + torch.exp(-x))


class SoftMax(Module):
    def forward(self, x: Tensor) -> Tensor:
        exp = torch.exp(x)
        exp_sum = exp.sum(dim=1, keepdim=True)
        return exp / exp_sum


class CrossEntropyLoss():
    def __call__(self, x, t) -> torch.Any:
        return self.forward(x, t)
    
    def forward(self, x: Tensor, t: Tensor):
        log_x = torch.log(x + 1e-8)
        return -torch.mean(t.unsqueeze(-2) @ log_x.unsqueeze(-1))