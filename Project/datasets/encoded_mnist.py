import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import numpy as np


class EncodedMNIST(Dataset):
    """
    A dataset class for MNIST that encodes images and labels beforehand using
    provided autoencoder and embedding models.

    Parameters:
        autoencoder (nn.Module): The autoencoder model for image encoding.
        embedding_model (nn.Module): The model to encode labels.
        root (str): Root directory of MNIST dataset.
        train (bool): If True, create dataset from training set, else from test set.
        transform (callable, optional): Optional transform to be applied on a PIL image.
    """

    def __init__(self, autoencoder, embedding_model, root="./datasets", train=True):
        self.mnist = MNIST(root=root, train=train,
                           transform=transforms.ToTensor(), download=True)
        self.autoencoder = autoencoder
        self.embedding_model = embedding_model
        self.encoded_data = []
        self._encode_dataset()

    def _encode_dataset(self):
        data_loader = DataLoader(self.mnist, batch_size=500, shuffle=False)
        for imgs, labels in tqdm(data_loader, desc='Encoding'):
            encoded_imgs = self.autoencoder.encoder(imgs).detach()
            labels = self._convert_labels(labels)
            encoded_labels = self.embedding_model.encode(
                labels, convert_to_tensor=True).detach()
            self.encoded_data.extend(zip(encoded_imgs, encoded_labels))

    def _convert_labels(self, labels, seed=42):
        """Converts labels from integers to strings in various languages (vectorized).

        Args:
            labels: A list of integer labels.
            seed: An integer to seed the random number generator for reproducibility.

        Returns:
            A list of strings where 25% of the labels are the original integer as a string,
            25% are written in English, 25% in German, and 25% in Spanish.
        """

        # Seed the random number generator for reproducability
        random.seed(seed)

        # Try converting labels to integers, handle non-convertible elements
        try:
            labels = [int(x) for x in labels]
        except ValueError:
            raise ValueError(
                "List elements must be integers or convertible to integers.")

        # Ensure labels are within the valid range (0-9)
        if any(x < 0 or x > 9 for x in labels):
            raise ValueError("List elements must be between 0 and 9.")

        # Define language probabilities
        language_probs = np.array([0.25, 0.25, 0.25, 0.25])

        print(len(labels))
        size = len(labels)
        # Sample random languages based on probabilities
        languages = np.random.choice(
            ["string", "english", "german", "spanish"], size=len(labels), p=language_probs)

        # Convert labels based on languages (vectorized)
        label_strings = np.where(
            languages == "string",
            str(labels),
            np.where(
                languages == "english",
                np.array(["zero", "one", "two", "three", "four",
                         "five", "six", "seven", "eight", "nine"])[labels],
                np.where(
                    languages == "german",
                    np.array(["null", "eins", "zwei", "drei", "vier",
                             "fünf", "sechs", "sieben", "acht", "neun"])[labels],
                    np.array(["cero", "uno", "dos", "tres", "cuatro",
                             "cinco", "seis", "siete", "ocho", "nueve"])[labels],
                ),
            ),
        )

        return label_strings.tolist()

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        encoded_img, encoded_label = self.encoded_data[idx]
        return encoded_img, encoded_label


class DynamicEncodedMNIST(Dataset):
    """
    A dataset class for MNIST that encodes images and labels on demand using
    provided autoencoder and embedding models.

    Parameters:
        autoencoder (nn.Module): The autoencoder model for image encoding.
        embedding_model (nn.Module): The model to encode labels.
        root (str): Root directory of MNIST dataset.
        train (bool): If True, create dataset from training set, else from test set.
        transform (callable, optional): Optional transform to be applied on a PIL image.
    """

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
            encoded_image = self.autoencoder.encoder(image).detach()

        # embedded input
        with torch.no_grad():
            encoded_target = self.embedder.encode(
                str(target), convert_to_tensor=True).detach()

        return encoded_image, encoded_target
