import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
import wandb


from datasets import EncodedMaskedMNIST
from cINN import ConditionalRealNVPImageTranslator
from autoencoder import AutoencoderSimple
from classifier import CNN
from utils import get_best_device

classifier_path = "./models/classifier_64.pth"
classifier = CNN()
classifier.load_state_dict(torch.load(
    classifier_path, map_location=get_best_device()))

ae_path = "./models/ae_100_new.pth"
autoencoder = AutoencoderSimple()
autoencoder.load_state_dict(torch.load(
    ae_path, map_location=get_best_device()))

train_data = EncodedMaskedMNIST(
    autoencoder=autoencoder, classifier=classifier, train=True)

TRAIN_SIZE = int(0.95 * len(train_data))
VAL_SIZE = len(train_data) - TRAIN_SIZE

train_data, val_data = torch.utils.data.random_split(
    train_data, [TRAIN_SIZE, VAL_SIZE])

train_loader = DataLoader(train_data, batch_size=100,
                          shuffle=True)
val_loader = DataLoader(val_data, batch_size=100)

hidden_size = 128
n_blocks = 20
epochs = 40

cinn = ConditionalRealNVPImageTranslator(
    input_size=64, hidden_size=hidden_size, n_blocks=n_blocks, condition_size=64)

hyperparams = {
    "hidden_size": hidden_size,
    "n_blocks": n_blocks,
    "epochs": epochs
}

wandb_logger = WandbLogger(project="cINN-Image-Inpainting",
                           entity="network-to-network",
                           name="128_20b_inp64")  # optionally pass in run name with: name="test-run"
trainer = L.Trainer(max_epochs=epochs, logger=wandb_logger)
trainer.fit(model=cinn, train_dataloaders=train_loader,
            val_dataloaders=val_loader)

torch.save(cinn.state_dict(), 'cinn_inpainting_40_inp64.pth')
