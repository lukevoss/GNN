import torch
import torch.nn as nn
import torch.nn.functional as F


class OurConditionalRealNVP(nn.Module):
    def __init__(self, input_size, hidden_size, condition_size, blocks):
        """
        Initialize a ConditionalRealNVP model.

        Args:
        - input_size (int): Total size of the input data.
        - hidden_size (int): Size of the hidden layers in the neural networks.
        - condition_size (int): Size of the condition vector (e.g., one-hot encoded label size).
        - blocks (int): Number of coupling layers in the model.
        """
        super(OurConditionalRealNVP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.condition_size = condition_size
        self.blocks = blocks

        # List of coupling layers
        self.coupling_layers = nn.ModuleList(
            [
                ConditionalCouplingLayer(
                    input_size, hidden_size, condition_size)
                for _ in range(blocks)
            ]
        )

        # List to store orthonormal matrices
        self.orthonormal_matrices = [
            self._get_orthonormal_matrix(input_size) for _ in range(blocks)
        ]

        # List to store scaling_before_exp for each block
        self.scaling_before_exp_list = []

    def _get_orthonormal_matrix(self, size):
        """
        Generate a random orthonormal matrix.

        Args:
        - size (int): Size of the matrix.

        Returns:
        - q (torch.Tensor): Orthonormal matrix.
        """
        w = torch.randn(size, size)
        q, _ = torch.linalg.qr(w, "reduced")
        return q

    def forward_realnvp(self, x, condition):
        """
        Forward pass through the ConditionalRealNVP model.

        Args:
        - x (torch.Tensor): Input data.
        - condition (torch.Tensor): Condition vector.

        Returns:
        - x (torch.Tensor): Transformed data.
        """
        scaling_before_exp_list = []
        for i in range(self.blocks):
            # print("x is:"); print(x)
            # print("shape of x is:"); print(x.shape)
            x = torch.matmul(x, self.orthonormal_matrices[i])
            x, scaling_before_exp = self.coupling_layers[i].forward(
                x, condition)
            scaling_before_exp_list.append(scaling_before_exp)

        self.scaling_before_exp_list = scaling_before_exp_list
        return x

    def decode(self, z, condition):
        """
        Reverse transformations to decode data.

        Args:
        - z (torch.Tensor): Transformed data.
        - condition (torch.Tensor): Condition vector.

        Returns:
        - z (torch.Tensor): Reconstructed original data.
        """
        for i in reversed(range(self.blocks)):
            z = self.coupling_layers[i].backward(z, condition)
            z = torch.matmul(z, self.orthonormal_matrices[i].t())
        return z

    def sample(self, num_samples=1000, conditions=None):
        """
        Generate synthetic samples.

        Args:
        - num_samples (int): Number of synthetic samples to generate.
        - conditions (torch.Tensor): Conditions for generating synthetic samples.

        Returns:
        - synthetic_samples (torch.Tensor): Synthetic samples.
        """
        with torch.no_grad():
            z = torch.randn(num_samples, self.input_size)
            synthetic_samples = self.decode(z, conditions)
        return synthetic_samples


class ConditionalCouplingLayer(nn.Module):
    def __init__(self, input_size, hidden_size, condition_size):
        """
        Initialize a ConditionalCouplingLayer.

        Args:
        - input_size (int): Total size of the input data.
        - hidden_size (int): Size of the hidden layers in the neural networks.
        - condition_size (int): Size of the condition vector (e.g., one-hot encoded label size).
        """
        super(ConditionalCouplingLayer, self).__init__()
        # Neural networks for the first half of the dimensions
        self.fc1 = nn.Linear(input_size // 2 + condition_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Translation coefficient
        self.fc3 = nn.Linear(hidden_size, input_size // 2)
        # Scaling coefficient
        self.fc4 = nn.Linear(hidden_size, input_size // 2)

    def forward(self, x, condition):
        """
        Forward pass through the ConditionalCouplingLayer.

        Args:
        - x (torch.Tensor): Input data.
        - condition (torch.Tensor): Condition vector.

        Returns:
        - y (torch.Tensor): Transformed data.
        - scaling_before_exp (torch.Tensor): Scaling coefficients before the exponential operation.
        """
        # Split the input into two halves
        x_a, x_b = x.chunk(2, dim=1)

        # Concatenate conditions to the first half
        x_a_concat = torch.cat([x_a, condition], dim=1)

        # Apply neural network to calculate coefficients
        h = F.relu(self.fc1(x_a_concat))
        h = F.relu(self.fc2(h))
        translation = self.fc3(h)
        scaling_before_exp = torch.tanh(self.fc4(h))
        scaling = torch.exp(scaling_before_exp)

        # Apply the affine transformation
        y_b = x_b * scaling + translation

        # Concatenate the transformed halves
        y = torch.cat([x_a, y_b], dim=1)
        return y, scaling_before_exp

    def backward(self, y, condition):
        """
        Backward pass through the ConditionalCouplingLayer.

        Args:
        - y (torch.Tensor): Transformed data.
        - condition (torch.Tensor): Condition vector.

        Returns:
        - x (torch.Tensor): Reconstructed original input.
        """
        # Split the input into two halves
        y_a, y_b = y.chunk(2, dim=1)

        # Concatenate conditions to the first half
        y_a_concat = torch.cat([y_a, condition], dim=1)

        # Apply neural network to calculate coefficients (reverse)
        h = F.relu(self.fc1(y_a_concat))
        h = F.relu(self.fc2(h))
        translation = self.fc3(h)
        scaling_before_exp = self.fc4(h)
        scaling = torch.exp(torch.tanh(scaling_before_exp))

        # Reverse the operations to reconstruct the original input
        x_a = y_a
        x_b = (y_b - translation) / scaling

        # Concatenate the reconstructed halves
        x = torch.cat([x_a, x_b], dim=1)
        return x
