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

        with torch.no_grad():  # Ensure no gradients are computed for the autoencoder
            image = image.unsqueeze(0)  # Add batch dimension
            encoded_image = self.autoencoder.encoder(image).squeeze(0)  # Remove batch dimension after encoding

        with torch.no_grad():  # Ensure no gradients are computed for the embedder
            target = torch.tensor(target)  # Convert target to tensor if necessary
            encoded_target = self.embedder(target.unsqueeze(0)).squeeze(0)  # Assuming embedder expects a batch dimension

        return encoded_image, encoded_target
