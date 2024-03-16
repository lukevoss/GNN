from .custom_mnist import CustomMNIST
from .encoded_mnist import EncodedMNIST, DynamicEncodedMNIST
from .masked_mnist import EncodedMaskedMNIST, MaskedMNIST
from .superresolution_mnist import SuperresolutionMNIST

__all__ = ['CustomMNIST',
           'EncodedMNIST',
           'DynamicEncodedMNIST',
           'EncodedMaskedMNIST',
           'MaskedMNIST',
           'SuperresolutionMNIST']
