import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset
import torch


class MixedMNIST(Dataset):
    def __init__(self,colored_rgb_mnist_dataset,pre_trained_ae,
                    root="./datasets_bla_bla", train=True ):
        """
        mix of uncolored (expanded accordingly) and colored mnist dataset
        Parameter:
            root (str): Root directory of dataset where ``MNIST/processed/training.pt``
                        and  ``MNIST/processed/test.pt`` will be saved.
            train (bool):   If True, creates dataset from ``training.pt``,
                            otherwise from ``test.pt``.
        """
        # Define transformations to apply to the data
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.mnist = MNIST(root=root, train=train,
                            transform=transform, download=True)
        
        self.mnist_expanded_images=self._expand_and_normalise()# it would be a torch tensor
        self.colored_mnist=colored_rgb_mnist_dataset # it would also be a torch tensor
        self.ae_model= pre_trained_ae

        # concatenate images
        self.mixed_images= self._concatenate()
        self.mixed_images_labels= self._labels()

    
    def _expand_and_normalise(self):
        image_data=self.mnist.data
        image_data= image_data/255.
        resized= image_data.view(-1,28,28,1) 
        expand= resized.expand(-1,-1,-1,3) # Assuming the last dimension is the channel dimension
        return expand
    
    def _concatenate(self):
        concat= torch.cat((self.mnist_expanded_images, self.colored_mnist), dim=0)
        return concat
    
    def _labels(self):
        len_colored_data=self.colored_mnist.size()[0]
        labels= torch.cat((torch.zeros(len_colored_data//2),torch.ones(len_colored_data//2)))
        labels= labels.unsqueeze(1)
        return labels

    def __len__(self):
        """
        Return the number of samples in concatenated dataset.
        """
        return len(self.mixed_images)

    def __getitem__(self, idx):
        """
        Fetch a sample for a given index.
        """
        return self.mixed_images[idx], self.mixed_images_labels[idx]
    

def encodings_inputs_for_style_transfer(ae_model, data):
        ae_model.eval()
        encodings = ae_model.encoder(data.permute(0, 3, 1, 2))
        return encodings
    

### Create custom dataset for encodings
### use this class if you have already encoded the data
class CustomEncodedStyleTransfer(Dataset):
    def __init__(self, data, labels):
        self.encoded_data = data
        self.color_labels = labels  # 0 for not colored, 1 for colored

    def __len__(self):
        return len(self.encoded_data)

    def __getitem__(self, idx):
        return self.encoded_data[idx], self.color_labels[idx]




