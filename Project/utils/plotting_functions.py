import matplotlib.pyplot as plt


def plot_mnist_samples(images, labels, captions, rows=2, cols=4):
    fig, axes = plt.subplots(rows, cols, figsize=(10, 5))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap="gray")
        ax.set_title(captions[i])
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_reconstruction_comparison(original_images, reconstructed_images, num_images=8):
    """
    Plots original and reconstructed images side by side.
    :param original_images: tensor of original images.
    :param reconstructed_images: tensor of reconstructed images.
    :param num_images: number of images to display.
    """
    fig, axes = plt.subplots(
        nrows=2, ncols=num_images, figsize=(2 * num_images, 4), sharex=True, sharey=True
    )
    for i in range(num_images):
        axes[0, i].imshow(original_images[i].squeeze(), cmap="gray")
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        axes[1, i].imshow(reconstructed_images[i].squeeze(), cmap="gray")
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis("off")
    plt.show()
