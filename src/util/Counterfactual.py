import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
from util import  DEVICE, IMAGE_SIZE, MEAN, STD, VAL_TRANSFORMATION
from PIL import Image
import numpy as np


# Utility functions for normalization and denormalization
def denormalize(image):
    mean = torch.tensor(MEAN, device=DEVICE).view(1, -1, 1, 1)
    std = torch.tensor(STD, device=DEVICE).view(1, -1, 1, 1)
    return image * std + mean

def normalize(image):
    mean = torch.tensor(MEAN, device=DEVICE).view(1, -1, 1, 1)
    std = torch.tensor(STD, device=DEVICE).view(1, -1, 1, 1)
    return (image - mean) / std

def clip_image_values(image):
    return torch.clamp(image, 0, 1)



def plot_histograms(original, counterfactual, title="Histogram Analysis"):
    """
    Plots histograms for the original and counterfactual images.
    """
    # Denormalize images
    original = denormalize(original).squeeze().detach().cpu().numpy()
    counterfactual = denormalize(counterfactual).squeeze().detach().cpu().numpy()

    # Flatten the images to 1D arrays for histogram computation
    original_flat = original.ravel()
    counterfactual_flat = counterfactual.ravel()

    # Plot histograms
    plt.figure(figsize=(12, 6))
    plt.hist(original_flat, bins=50, alpha=0.8, color='blue', label='Original Image')
    plt.hist(counterfactual_flat, bins=50, alpha=0.3, color='red', label='Counterfactual Image')
    plt.title(title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def plot_difference_histogram(original, counterfactual, title="Difference Histogram"):
    """
    Plots the histogram of pixel-wise differences between the original and counterfactual images.
    """
    # Denormalize images
    original = denormalize(original).squeeze().detach().cpu().numpy()
    counterfactual = denormalize(counterfactual).squeeze().detach().cpu().numpy()

    # Compute the difference
    difference = counterfactual - original
    difference_flat = difference.ravel()

    # Plot histogram of differences
    plt.figure(figsize=(12, 6))
    plt.hist(difference_flat, bins=50, alpha=0.7, color='green', label='Pixel Differences')
    plt.title(title)
    plt.xlabel('Pixel Intensity Difference')
    plt.ylabel('Frequency')
    plt.axvline(0, color='black', linestyle='dashed', linewidth=1, label='No Change')
    plt.legend()
    plt.show()

def counterfactual(model, image_path, target_class=None, alpha=10.0, beta=0.01, num_steps=200):
    """
    Prediction Loss (alpha): 
    This term ensures that the counterfactual image is classified as the target class by the model.
    It measures how well the counterfactual image achieves the desired output (e.g., being classified as a dog instead of a cat).

    Proximity Loss (beta):    
    This term ensures that the counterfactual image remains close to the original image.
    It measures how much the counterfactual image differs from the original image in terms of pixel values or features.
    """
  
    if target_class is None:
        raise ValueError("Target class must be provided for counterfactual generation.")

    model.eval()
    model.to(DEVICE)

    # Load and preprocess the image
    transform = VAL_TRANSFORMATION
    original_image = Image.open(image_path).convert("RGB")
    original_image = transform(original_image).unsqueeze(0).to(DEVICE)

    # Wrap input image in a parameter for optimization
    input_image = original_image.clone().detach().requires_grad_(True)
    input_image = nn.Parameter(input_image)

    # Define loss components
    prediction_loss = nn.CrossEntropyLoss()
    def proximity_loss(x, original):
        return torch.norm(denormalize(x) - denormalize(original))

    # Optimizer
    optimizer = optim.Adam([input_image], lr=0.001)

    # Generate counterfactual
    for step in range(num_steps):
        optimizer.zero_grad()

        # Forward pass
        output = model(input_image)
        target_score = prediction_loss(output, torch.tensor([target_class], device=DEVICE))

        # Calculate proximity loss
        proximity = proximity_loss(input_image, original_image)

        # Combined loss
        loss = alpha * target_score + beta * proximity
        loss.backward()

        # Gradient clipping (optional)
       # torch.nn.utils.clip_grad_norm_(input_image, max_norm=1.0)

        # Update the counterfactual image
        optimizer.step()

        # Clip to valid image range
        input_image.data = clip_image_values(denormalize(input_image))
        input_image.data = normalize(input_image.data)

        # Print progress with confidence score
        confidence = torch.softmax(output.detach(), dim=1)[0, target_class].item()
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}, Confidence for target class: {confidence:.4f}")

    # Visualization functions
    def show_image(img, title):
        img = denormalize(img).squeeze().detach().permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        plt.show()

    def show_difference(original, modified):
        diff = (denormalize(modified) - denormalize(original)).squeeze().detach().cpu().permute(1, 2, 0).numpy()
        max_diff = max(abs(diff.min()), abs(diff.max()))
        norm_diff = diff / max_diff  
        plt.figure(figsize=(8, 8))
        img = plt.imshow(norm_diff, cmap='seismic', vmin=-1, vmax=1)
        plt.colorbar(img, orientation='vertical', label='Difference Intensity')
        plt.title("Difference (Counterfactual - Original)")
        plt.axis('off')
        plt.figtext(0.5, 0.02, 
                   f"Blue = Features Suppressed | Red = Features Enhanced | Range: [{-max_diff:.2f}, {max_diff:.2f}]",
                    ha='center', fontsize=10, wrap=True)
        plt.show()

    # Visualize results
    show_image(original_image, "Original Image")
    show_image(input_image, "Counterfactual Image")
    show_difference(original_image, input_image)

     # Histogram analysis
    plot_histograms(original_image, input_image, title="Histogram Analysis: Original vs Counterfactual")
    plot_difference_histogram(original_image, input_image, title="Histogram of Pixel Differences")

    