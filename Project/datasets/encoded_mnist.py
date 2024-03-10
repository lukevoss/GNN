import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset


class EncodedMNIST(Dataset):
    def __init__(self, autoencoder, embedder, root="./datasets", train=True):
        """
        Create encoded MNIST dataset to translate between embedding and latent space representation
        """
        # Define transformations to apply to the data
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]
        )
        self.mnist = MNIST(root=root, train=train,
                           transform=transform, download=True)
        self.autoencoder = autoencoder
        self.embedder = embedder
        self.autoencoder.eval()
        self.embedder.eval()

    def __len__(self):
        """
        Return the number of samples in dataset.
        """
        return len(self.mnist)


    def __getitem__(self, idx):
        """
        Fetch a sample for a given index, applying the autoencoder and embedder transformations.
        """
        image, target = self.mnist[idx]

        # latent space representation
        with torch.no_grad():
            encoded_image = self.autoencoder.encoder(image)

        # embedded input
        with torch.no_grad():  
            encoded_target = self.embedder.encode(str(target), convert_to_tensor=True)

        return encoded_image, encoded_target
