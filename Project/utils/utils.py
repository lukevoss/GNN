import torch

from .plotting_functions import plot_reconstruction_comparison


def get_best_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device("cpu")


def test_autoencoder(autoencoder, test_loader, device=get_best_device()):
    """
    Tests the autoencoder on a batch from the test_loader and plots original and reconstructed images.
    :param autoencoder: The autoencoder model.
    :param test_loader: DataLoader for the test dataset.
    :param device: The device to run the model on.
    """
    autoencoder.eval()

    # Get a batch of test images
    images, _ = next(iter(test_loader))
    images = images.to(device)

    # Reconstruct images using the autoencoder
    with torch.no_grad():
        reconstructed_images = autoencoder(images)

    # Prepare images for display
    original_images = images.cpu()
    reconstructed_images = reconstructed_images.cpu()

    # Plot original and reconstructed images
    plot_reconstruction_comparison(original_images, reconstructed_images)
