import torch
from torch import nn
import lightning as L


class AutoencoderSimple(L.LightningModule):
    def __init__(self):
        super(AutoencoderSimple, self).__init__()

        # Encoder layers
        self.encoder_conv1 = nn.Conv2d(
            1, 6, kernel_size=3, stride=2, padding=1
        )  # Use stride 2 for downscaling
        self.encoder_conv2 = nn.Conv2d(
            6, 12, kernel_size=3, stride=2, padding=1
        )  # Use stride 2 for downscaling
        self.encoder_linear = nn.Linear(12 * 7 * 7, 64)  # 64*6 ---> 20

        # Decoder layers
        self.decoder_linear = nn.Linear(64, 12 * 7 * 7)
        self.decoder_conv1 = nn.ConvTranspose2d(
            12, 6, kernel_size=3, stride=2, padding=1, output_padding=1
        )  # Use stride 2 for upscaling
        self.decoder_conv2 = nn.ConvTranspose2d(
            6, 1, kernel_size=3, stride=2, padding=1, output_padding=1
        )  # Use stride 2 for upscaling
        self.criterion = nn.MSELoss()

    def encoder(self, x):
        # Encoder
        x = self.encoder_conv1(x)
        x = torch.relu(x)
        x = self.encoder_conv2(x)
        x = torch.relu(x)
        x = x.view(-1, 12 * 7 * 7)
        x = self.encoder_linear(x)
        return x

    def decoder(self, x):
        # Decoder
        x = self.decoder_linear(x)
        x = x.view(-1, 12, 7, 7)
        x = self.decoder_conv1(x)
        x = torch.relu(x)
        x = self.decoder_conv2(x)
        x = torch.relu(x)
        return x

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, _ = batch
        outputs = self(images)
        loss = self.criterion(outputs, images)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, _ = batch
        outputs = self(images)
        loss = self.criterion(outputs, images)
        self.log("val_loss", loss)
