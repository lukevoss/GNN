import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset


class CustomMNIST(Dataset):
    def __init__(self, root="./datasets", train=True):
        """
        Initialize normalized MNIST dataset in Tensor format.

        Parameter:
            root (str): Root directory of dataset where ``MNIST/processed/training.pt``
                        and  ``MNIST/processed/test.pt`` will be saved.
            train (bool):   If True, creates dataset from ``training.pt``,
                            otherwise from ``test.pt``.
        """
        self.mnist = MNIST(root=root, train=train,
                           transform=transforms.ToTensor(), download=True)

    def __len__(self):
        """
        Return the number of samples in dataset.
        """
        return len(self.mnist)

    def __getitem__(self, idx):
        """
        Fetch a sample for a given index.
        """
        return self.mnist[idx]
