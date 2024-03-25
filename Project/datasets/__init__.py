from .custom_mnist import CustomMNIST
from .encoded_mnist import EncodedMNIST, DynamicEncodedMNIST
from .masked_mnist import EncodedMaskedMNIST, MaskedMNIST
from .custom_encoding_style_transfer import MixedMNIST, CustomEncodedStyleTransfer
from .superresolution_mnist import (
    SuperresolutionMNIST,
    EncodedSuperresolutionMNIST,
    SuperresolutionEMNIST,
)

__all__ = [
    "CustomMNIST",
    "EncodedMNIST",
    "DynamicEncodedMNIST",
    "EncodedMaskedMNIST",
    "MaskedMNIST",
    "SuperresolutionMNIST",
    "EncodedSuperresolutionMNIST",
    "MixedMNIST",
    "CustomEncodedStyleTransfer",
    "SuperresolutionEMNIST",
]
