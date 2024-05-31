import torch
from torch import Tensor
import matplotlib.pyplot as plt


class Module:
    def params(self) -> list[tuple[Tensor, Tensor]]:
        """return an iterator of parameters to be updated"""
        return []


class Layer(Module):
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError


class Loss(Module):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        raise NotImplementedError

    def backward(self) -> Tensor:
        raise NotImplementedError


class Linear(Layer):
    w: Tensor  # weight
    b: Tensor  # bias
    x: Tensor  # x from the last forward, to be used in backprop
    dw: Tensor  # gradient for W
    db: Tensor  # gradient for b

    def __init__(self, num_prev: int, num: int) -> None:
        """generate an Affine layer with `num` neurons, connected to `num_prev` 
        neurons in the previous layer

        Args:
            num_prev (int): number of neurons in the previous layer
            num (int): number of neurons in this layer
        """
        self.w = 0.1 * torch.randn(num_prev, num)  # random matrix of shape (num_prev, num)
        self.b = 0.1 * torch.randn(1, num)  # random weight of shape (1, num)

    def forward(self, x: Tensor) -> Tensor:
        """ Part of the forward pass of the network, computing the output of this layer
        formula: out = x @ w + b | (n, c) x (c, d) + (..., d) = (n, d)
        """
        # print(x.shape)
        # print(self.w.shape)
        assert x.shape[1] == self.w.shape[0]
        self.x = x
        return x @ self.w + self.b

    def backward(self, dl: Tensor) -> Tensor:
        """Part of the backward pass of the network, given dl from the next layer, compute dw and db
        formula: dw = dl @ x.T         | (n, c) @ (c, d) = (n, d)
                 db_j = sum_i(dl_ij)   | (1, d)
                 dx = w.T @ dl.T       | (c, d) @ (d, n) = (c, n)
        """
        assert dl.shape[1] == self.w.shape[1]
        self.dw = self.x.T @ dl
        self.db = dl.sum(0, keepdim=True)
        return dl @ self.w.T

    def params(self):
        return (self.w, self.dw), (self.b, self.db)


class Sigmoid(Layer):
    y: Tensor | None

    def __init__(self):
        self.y = None

    def forward(self, x: Tensor) -> Tensor:
        self.y = 1 / (1 + torch.exp(-x))
        return self.y

    def backward(self, dl: Tensor) -> Tensor:
        """"""
        return dl * self.y * (1 - self.y)


class Softmax(Layer):
    """Softmax layer"""
    m: Tensor | None
    dim: int | None

    def __init__(self):
        self.m = None

    def forward(self, x: Tensor) -> Tensor:
        """x is the input tensor, t is the expected output tensor
        formula: softmax(x)_i = exp(x_i)/sum_j(exp(x_j))
        """
        exp = torch.exp(x)
        exp_sum = torch.sum(exp, dim=1, keepdim=True)
        self.m = exp / exp_sum
        return self.m

    def backward(self, dl: Tensor) -> Tensor:
        """compute backward pass for softmax
        formula: for each row x,
        m = softmax(x),
        J = -m.T @ m + I*m
        dx = dl @ J
        """
        # adding one layer in between to transpose
        # m: [n, dim, 1]
        # m.T: [n, 1, dim]
        m = self.m.unsqueeze(-1)
        mt = self.m.unsqueeze(-2)
        j = -(m @ mt) + m * torch.eye(self.m.shape[1])
        return torch.einsum("ij,ijk->ik", dl, j)


class Relu(Layer):
    mask: Tensor | None

    def __init__(self):
        self.mask = None

    def forward(self, x: Tensor) -> Tensor:
        self.mask = (x <= 0)
        out = x.clone()
        out[self.mask] = 0
        return out

    def backward(self, dl: Tensor) -> Tensor:
        dl[self.mask] = 0
        return dl


class CrossEntropyLoss(Loss):
    t: Tensor | None
    x: Tensor | None
    y: Tensor | None

    def forward(self, x: Tensor, t: Tensor = None) -> Tensor:
        if t is None:
            raise ValueError("target not provided")
        self.t = t
        self.x = x
        log_x = torch.log(x + 1e-8)
        return -torch.mean(t.unsqueeze(-2) @ log_x.unsqueeze(-1))

    def backward(self, dl: Tensor = None) -> Tensor:
        if len(self.t.shape) == 2:
            self.t = torch.argmax(self.t, dim=1)

        dx = self.x.clone()
        dx[range(len(dx)), self.t] -= 1
        dx /= self.t.shape[0]
        return dx


class Optimizer:
    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, layers: list[Layer], lr: float) -> None:
        self.layers = layers
        self.lr = lr

    def step(self):
        for layer in self.layers:
            for (param, grad) in layer.params():
                # assert param.shape == grad.shape
                # print(param.shape)
                # print(grad.shape)
                param -= self.lr * grad


class MLP:
    def __init__(self, layers: list[Layer], loss: Loss) -> None:
        self.layers = layers
        self.loss = loss

    def forward(self, x: Tensor) -> Tensor:
        x = x.clone()
        pass
        for layer in self.layers:
            x = layer.forward(x)
            pass
        return x

    def backward(self) -> None:
        y = self.loss.backward()
        for layer in self.layers[::-1]:
            y = layer.backward(y)

    def params(self) -> list[tuple[Tensor, Tensor]]:
        params = []
        for layer in self.layers:
            params += layer.params()
        return params

    def train(self, x_train: Tensor, t_train: Tensor, x_val: Tensor, t_val: Tensor,
              batch_size: int, num_epochs: int,
              lr: float = 0.05) -> None:
        if len(t_val.shape) > 1:
            t_val = torch.argmax(t_val, dim=1)

        optimizer = SGD(self.layers, lr)
        train_losses = []
        valid_acc = []

        for i in range(num_epochs):
            print(f"epoch {i}")
            indices = torch.randperm(len(x_train))
            for j in range(0, len(indices), batch_size):
                batch_ind = indices[j: j + batch_size]
                x_batch = x_train[batch_ind]
                t_batch = t_train[batch_ind]
                y_batch = self.forward(x_batch)
                loss = self.loss.forward(y_batch, t_batch)
                train_losses.append(loss.item())
                self.backward()
                optimizer.step()
            y_val = torch.argmax(self.forward(x_val), dim=1)
            valid_acc.append((y_val == t_val).sum().item() / len(t_val))
        plt.plot(train_losses)
        plt.title(f"train loss, batch_size={batch_size}, num_epoch={num_epochs}, lr={lr}")
        plt.show()
        plt.plot(valid_acc)
        plt.title(f'validation accuracy, batch_size={batch_size}, num_epoch={num_epochs}, lr={lr}')
        plt.show()
