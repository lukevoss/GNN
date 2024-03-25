import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
import numpy as np


class MaskedMNIST(Dataset):
    def __init__(self, root="./datasets", train=True):
        """
        Initialize MNIST with randomply masked areas in rectengular format

        Parameter:
            root (str): Root directory of dataset where ``MNIST/processed/training.pt``
                        and  ``MNIST/processed/test.pt`` will be saved.
            train (bool):   If True, creates dataset from ``training.pt``,
                            otherwise from ``test.pt``.
        """
        self.mnist = MNIST(
            root=root, train=train, transform=transforms.ToTensor(), download=True
        )

    def __len__(self):
        """
        Return the number of samples in dataset.
        """
        return len(self.mnist)

    def __getitem__(self, idx):
        """
        Fetch a sample for a given index, apply a random squared mask, and return both masked and original images.
        """
        image, label = self.mnist[idx]

        # Generate a random square mask
        masked_image, mask = self.apply_random_mask(image)

        # Return the masked image, original image, and label
        return image, masked_image, label

    def apply_random_mask(self, image):
        """
        Apply a random square mask to the image.
        """
        im_size = image.size(1)
        # Random mask size between 1/4 to 1/2 of image size
        mask_size = np.random.randint(im_size // 4, im_size // 2)
        top = np.random.randint(0, im_size - mask_size)
        left = np.random.randint(0, im_size - mask_size)

        mask = torch.ones_like(image)
        mask[:, top : top + mask_size, left : left + mask_size] = 0  # Apply mask

        masked_image = image.clone()  # Clone to not modify the original image
        masked_image[:, top : top + mask_size, left : left + mask_size] = -1

        return masked_image, mask


class EncodedMaskedMNIST(Dataset):
    def __init__(self, classifier, autoencoder, root="./datasets", train=True):
        self.classifier = classifier
        self.autoencoder = autoencoder
        self.classifier.eval()
        self.autoencoder.eval()
        self.masked_mnist = MaskedMNIST(root=root, train=train)

    def __len__(self):
        """
        Return the number of samples in dataset.
        """
        return len(self.masked_mnist)

    def __getitem__(self, idx):
        image, masked_image, label = self.masked_mnist[idx]

        with torch.no_grad():
            encoded_masked_image = self.classifier.encode(masked_image).detach()

        with torch.no_grad():
            encoded_image = self.autoencoder.encoder(image).detach()

        # Return the masked image, original image, and label
        return encoded_image.squeeze(0), encoded_masked_image.squeeze(0), label
