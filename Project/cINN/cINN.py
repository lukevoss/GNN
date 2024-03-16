import torch
import torch.nn as nn
import math
from scipy.stats import special_ortho_group
import lightning as L
import numpy as np


class ConditionalRealNVP(L.LightningModule):
    def __init__(self, input_size, hidden_size, n_blocks, condition_size, learning_rate=1e-3):
        """
        Initialize a ConditionalRealNVP model.

        Args:
        - input_size (int): Total size of the input data.
        - hidden_size (int): Size of the hidden layers in the neural networks.
        - condition_size (int): Size of the condition vector (e.g., one-hot encoded label size).
        - n_blocks (int): Number of coupling layers in the model.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_blocks = n_blocks
        self.condition_size = condition_size
        self.learning_rate = learning_rate
        self.coupling_blocks = nn.ModuleList(
            [
                ConditionalCouplingBlock(
                    input_size, hidden_size, self.condition_size)
                for _ in range(n_blocks)
            ]
        )
        self.orthogonal_matrices = nn.ParameterList([
            nn.Parameter(self._create_orthogonal_matrix(
                input_size), requires_grad=False)
            for _ in range(n_blocks - 1)
        ])

    def forward(self, x, cond, rev=False):
        if rev:
            return self._inverse(x, cond)
        return self._forward(x, cond)

    def _forward(self, x, cond):
        ljd = torch.zeros((x.shape[0]),  device=self.device)
        for l in range(self.n_blocks - 1):
            x, partial_ljd = self.coupling_blocks[l](x, cond)
            ljd += partial_ljd
            x = torch.matmul(x, self.orthogonal_matrices[l])
        x, partial_ljd = self.coupling_blocks[-1](x, cond)
        ljd += partial_ljd
        return x, ljd

    def _inverse(self, x, cond):
        for l in range(self.n_blocks - 1, 0, -1):
            x = self.coupling_blocks[l](x, cond, rev=True)
            x = torch.matmul(x, self.orthogonal_matrices[l - 1].T)
        x = self.coupling_blocks[0](x, cond, rev=True)
        return x

    def sample(self, num_samples, cond=None):
        samples = []
        # z = torch.normal(
        #     mean=torch.zeros((num_samples, self.input_size)),
        #     std=torch.ones((num_samples, self.input_size)),
        # )
        z = torch.randn(num_samples, self.input_size)
        if isinstance(cond, np.ndarray):
            cond = torch.from_numpy(cond)
        cond = cond.repeat(num_samples, 1)
        samples.append(self._inverse(z, cond=cond))
        return torch.cat(samples, 0)

    def _create_orthogonal_matrix(self, dim):
        Q = special_ortho_group.rvs(dim)
        return torch.tensor(Q, dtype=torch.float32, device=self.device)

    def training_step(self, batch, batch_idx):
        x_batch, cond_batch = batch
        z, ljd = self(x_batch, cond_batch)
        loss = torch.sum(0.5 * torch.sum(z**2, -1) - ljd) / x_batch.size(0)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_batch, cond_batch = batch
        z, ljd = self(x_batch, cond_batch)
        loss = torch.sum(0.5 * torch.sum(z**2, -1) - ljd) / x_batch.size(0)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class ConditionalCouplingBlock(L.LightningModule):
    def __init__(self, input_size, hidden_size, condition_size):
        """
        Initialize a ConditionalCouplingLayer.

        Args:
        - input_size (int): Total size of the input data.
        - hidden_size (int): Size of the hidden layers in the neural networks.
        - condition_size (int): Size of the condition vector (e.g., one-hot encoded label size).
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.condition_size = condition_size
        self.split1 = math.floor(self.input_size / 2)
        self.split2 = self.input_size - self.split1
        self.subnet = self._subnet_constructor(
            self.split1 + self.condition_size, self.hidden_size, 2 * self.split2
        )

    def forward(self, x, cond, rev=False):
        x1, x2 = x[..., : self.split1], x[..., self.split1:]
        params = self.subnet(torch.cat([x1, cond], -1))
        s, t = params[..., : self.split2], params[..., self.split2:]
        s = torch.tanh(s)
        ljd = torch.sum(s, -1)
        if not rev:
            s = torch.exp(s)
            x2 = s * x2 + t  # Apply the affine transformation
            return torch.cat([x1, x2], -1), ljd
        if rev:
            s = torch.exp(-s)
            x2 = s * (x2 - t)  # Reverse the affine transformation
            return torch.cat([x1, x2], -1)

    def _subnet_constructor(self, input_size, hidden_size, output_size):
        model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        return model
