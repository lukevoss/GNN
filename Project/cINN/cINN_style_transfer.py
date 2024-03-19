import torch.nn as nn
import torch
import lightning as L
from .cINN import ConditionalRealNVP, ConditionalCouplingBlock

class CustomConditionalCouplingBlock(ConditionalCouplingBlock):
    def _subnet_constructor(self, input_size, hidden_size, output_size
                            , dropout_rate=0.1):
        model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Adding dropout after ReLU
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Adding dropout after ReLU
            nn.Linear(hidden_size, output_size),
        )
        return model

class ConditionalRealNVPStyleTransfer(ConditionalRealNVP):
    def __init__(self, input_size, hidden_size, n_blocks, condition_size, learning_rate=1e-3):
        super().__init__(input_size, hidden_size, n_blocks, condition_size, learning_rate)

        # Override coupling_blocks initialization with CustomConditionalCouplingBlock
        self.coupling_blocks = nn.ModuleList(
            [
                CustomConditionalCouplingBlock(
                    input_size, hidden_size, self.condition_size)
                for _ in range(n_blocks)
            ]
        )