import matplotlib.pyplot as plt
import numpy as np

def plot_segmentation_mask(mask, class_names=None):
    """
    Plots a segmentation mask.

    Args:
        mask (numpy.ndarray): Segmentation mask (2D array of class indices).
        class_names (list): List of class names (optional).

    Returns:
        None
    """
    if class_names is None:
        class_names = [str(i) for i in range(mask.max() + 1)]

    plt.imshow(mask, cmap='tab20', vmin=0, vmax=len(class_names) - 1)
    plt.colorbar(ticks=np.arange(len(class_names)), label='Class')
    plt.xticks([])
    plt.yticks([])
    plt.title("Segmentation Mask")
    plt.show()



def plot_image_with_mask(image, mask, class_names=None):
    """
    Plots an input image with overlaid segmentation mask.

    Args:
        image (numpy.ndarray): Input image.
        mask (numpy.ndarray): Segmentation mask (2D array of class indices).
        class_names (list): List of class names (optional).

    Returns:
        None
    """
    plt.imshow(image)
    plt.imshow(mask, cmap='tab20', alpha=0.5, vmin=0, vmax=len(class_names) - 1)
    plt.xticks([])
    plt.yticks([])
    plt.title("Image with Segmentation Mask")
    plt.show()

def plot_predictions_vs_ground_truth(image, true_mask, pred_mask, class_names=None):
    """
    Plots input image, ground truth mask, and predicted mask side by side.

    Args:
        image (numpy.ndarray): Input image.
        true_mask (numpy.ndarray): Ground truth segmentation mask.
        pred_mask (numpy.ndarray): Predicted segmentation mask.
        class_names (list): List of class names (optional).

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    axes[1].imshow(true_mask, cmap='tab20', vmin=0, vmax=len(class_names) - 1)
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis('off')

    axes[2].imshow(pred_mask, cmap='tab20', vmin=0, vmax=len(class_names) - 1)
    axes[2].set_title("Predicted Mask")
    axes[2].axis('off')

    plt.show()


def plot_class_distribution(mask, class_names=None):
    """
    Plots the distribution of pixel counts for each class.

    Args:
        mask (numpy.ndarray): Segmentation mask (2D array of class indices).
        class_names (list): List of class names (optional).

    Returns:
        None
    """
    unique_classes, counts = np.unique(mask, return_counts=True)
    if class_names is None:
        class_names = [str(i) for i in unique_classes]

    plt.bar(class_names, counts)
    plt.xlabel("Class")
    plt.ylabel("Pixel Count")
    plt.title("Class Distribution")
    plt.show()
