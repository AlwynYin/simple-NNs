import torch
from torch import Tensor


class Layer():
    def feed_forward(self, x: Tensor) -> Tensor:
        """given input from previous layer, computer the output of this layer"""
        raise NotImplementedError

    def back_prop(self) -> Tensor:
        raise NotImplementedError


class Affine(Layer):
    W: Tensor   # weight
    b: Tensor   # bias
    x: Tensor   # x from the last forward, to be used in backprop
    dW: Tensor  # gradient for W
    db: Tensor  # gradient for b

    def __init__(self, num_prev: int, num: int) -> None:
        """generate an Affine layer with `num` neurons, connected to `num_prev` neurons in the previous layer

        Args:
            num_prev (int): number of neurons in the previous layer
            num (int): number of neurons in this layer
        """
        self.W = torch.randn(num, num_prev) # random matrix of shape (num, num_prev), mathematically num_prev x num
        self.b = torch.randn(num)           # random weight of num

    def forward(self, x: Tensor) -> Tensor:
        """ Part of the forward pass of the network, computing the output of this layer"""
        return torch.matmul(self.W, x) + self.b.unsqueeze(1)

    def backward(self, dl: Tensor) -> Tensor:
        """Part of the backward pass of the network, given dl from the next layer, compute dw and db"""
        self.dW = torch.dot(dl, self.x.T)
        self.db = dl.sum(1)
        return torch.dot(dl, self.W.T)


class Sigmoid(Layer):
    y: Tensor | None

    def __init__(self):
        self.y = None

    def forward(self, x: Tensor) -> Tensor:
        self.y = 1 / (1 + torch.exp(-x))
        return self.y

    def backward(self, dl: Tensor) -> Tensor:
        # sigmoid has no "weight", so it just changes the output and pass it off
        return dl * self.y * (1 - self.y)


class SoftmaxWithLoss(Layer):
    """Softmax layer with cross-entropy loss function"""
    t: Tensor | None
    y: Tensor | None

    def __init__(self, t: Tensor):
        """t is the expected output"""
        self.t = t

    def forward(self, x: Tensor) -> Tensor:
        """x is the input tensor, t is the expected output tensor"""
        exp = torch.exp(x)
        exp_sum = torch.sum(exp, 0)
        softmax = exp/exp_sum
        self.y = -torch.sum(self.t * torch.log(softmax))
        return self.y

    def backward(self, dl: Tensor) -> Tensor:
        size = self.t.shape[1]
        dx = (self.y - self.t) / size
        return dx


class Relu(Layer):
    mask: Tensor | None

    def __init__(self):
        self.mask = None

    def forward(self, x: Tensor) -> Tensor:
        self.mask = (x <= 0)
        out = x.detach().clone()
        out[self.mask] = 0
        return out

    def backward(self, dl: Tensor) -> Tensor:
        dl[self.mask] = 0
        return dl


class Network:
    layers: list[Layer]
    num_layers: int
