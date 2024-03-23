import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST, EMNIST
from torch.utils.data import Dataset


class SuperresolutionMNIST(Dataset):
    def __init__(self, root="./datasets", train=True):
        """
        Initialize normalized MNIST dataset in Tensor format.

        Parameter:
            root (str): Root directory of dataset where ``MNIST/processed/training.pt``
                        and  ``MNIST/processed/test.pt`` will be saved.
            train (bool):   If True, creates dataset from ``training.pt``,
                            otherwise from ``test.pt``.
        """
        reduce_size_transform = transforms.Compose([
            transforms.Resize((12, 12)),  # Train AE on mini Mnist
            transforms.ToTensor()
        ])

        self.mnist = MNIST(root=root, train=train,
                           transform=transforms.ToTensor(), download=True)
        self.small_mnist = MNIST(root=root, train=train,
                                 transform=reduce_size_transform, download=True)

    def __len__(self):
        """
        Return the number of samples in dataset.
        """
        return len(self.mnist)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]
        small_image, _ = self.small_mnist[idx]
        return image, small_image, label


class SuperresolutionEMNIST(Dataset):
    def __init__(self, root="./datasets", train=True, split="mnist"):
        """
        Initialize normalized MNIST dataset in Tensor format.

        Parameter:
            root (str): Root directory of dataset where ``MNIST/processed/training.pt``
                        and  ``MNIST/processed/test.pt`` will be saved.
            train (bool):   If True, creates dataset from ``training.pt``,
                            otherwise from ``test.pt``.
        """
        reduce_size_transform = transforms.Compose([
            transforms.Resize((12, 12)),  # Train AE on mini Mnist
            transforms.ToTensor()
        ])

        self.mnist = EMNIST(root=root, train=train,
                            transform=transforms.ToTensor(), download=True, split=split)
        self.small_mnist = EMNIST(root=root, train=train,
                                  transform=reduce_size_transform, download=True, split=split)

    def __len__(self):
        """
        Return the number of samples in dataset.
        """
        return len(self.mnist)

    def __getitem__(self, idx):
        image, label = self.mnist[idx]
        small_image, _ = self.small_mnist[idx]
        return image, small_image, label


class EncodedSuperresolutionMNIST(Dataset):
    def __init__(self, autoencoder_small, autoencoder_big, root="./datasets", train=True):
        self.autoencoder_small = autoencoder_small
        self.autoencoder_big = autoencoder_big
        self.autoencoder_small.eval()
        self.autoencoder_big.eval()
        self.superresolution_mnist = SuperresolutionEMNIST(
            root=root, train=train)

    def __len__(self):
        """
        Return the number of samples in dataset.
        """
        return len(self.superresolution_mnist)

    def __getitem__(self, idx):
        image, small_image, label = self.superresolution_mnist[idx]

        with torch.no_grad():
            encoded_small_image = self.autoencoder_small.encoder(
                small_image).detach()

        with torch.no_grad():
            encoded_image = self.autoencoder_big.encoder(image).detach()

        # Return the small image, original image, and label
        return encoded_image.squeeze(0), encoded_small_image.squeeze(0), label
