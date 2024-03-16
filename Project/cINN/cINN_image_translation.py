import torch
from .cINN import ConditionalRealNVP


class ConditionalRealNVPImageTranslator(ConditionalRealNVP):
    """
    This class inherits from our ConditionalRealNVP class but is adapted to process datasets for image translation
    containing image, translated_image, labels
    """

    def __init__(self, input_size, hidden_size, n_blocks, condition_size, learning_rate=1e-3):
        super().__init__(input_size, hidden_size, n_blocks, condition_size, learning_rate)

    def training_step(self, batch, batch_idx):
        x_batch, cond_batch, _ = batch
        z, ljd = self(x_batch, cond_batch)
        loss = torch.sum(0.5 * torch.sum(z**2, -1) - ljd) / x_batch.size(0)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_batch, cond_batch, _ = batch
        z, ljd = self(x_batch, cond_batch)
        loss = torch.sum(0.5 * torch.sum(z**2, -1) - ljd) / x_batch.size(0)
        self.log("val_loss", loss)
