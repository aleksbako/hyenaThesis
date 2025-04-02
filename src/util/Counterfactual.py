import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
from util import  DEVICE, IMAGE_SIZE, MEAN, STD, VAL_TRANSFORMATION
from PIL import Image
import numpy as np
from scipy import fftpack
from skimage.measure import label
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


def robustness_comparison(model_1,model_type_1,model_2,model_type_2, image_path, target_class, perturbation_sizes=[0.01, 0.05, 0.1]):
    """
    Measure how much perturbation is needed to change classification
    """
    robustness_data = []
    
    for eps in perturbation_sizes:
        row = {'perturbation_size': eps}
   
            # Find minimal perturbation needed
        cf_image, metrics = counterfactual(
                model_1,
                image_path,
                target_class=target_class,
                alpha=10.0,
                beta=0.1,  # Tighter proximity constraint
                num_steps=100
            )
            
        row[model_type_1] = {
                'l2_distance': metrics['l2_distance'],
                'confidence': metrics['final_confidence'],
                'success': metrics['success']
        }
        
        robustness_data.append(row)
    
    # Plot robustness curves
    plot_robustness_comparison(
        robustness_data,
        title="Perturbation Robustness Comparison"
    )

import seaborn as sns
from typing import Dict, List

def plot_robustness_comparison(
    robustness_data: List[Dict],
    title: str = "Robustness Comparison: Attention vs Hyena",
    save_path: str = None,
    figsize: tuple = (12, 6)
):
    """
    Plots comparison of robustness metrics between models across perturbation sizes.
    
    Args:
        robustness_data: List of dictionaries containing:
            [
                {
                    'perturbation_size': float,
                    'Attention': {'l2_distance': float, 'confidence': float, 'success': bool},
                    'Hyena': {'l2_distance': float, 'confidence': float, 'success': bool}
                },
                ...
            ]
        title: Plot title
        save_path: Path to save the figure (optional)
        figsize: Figure dimensions
    """
    plt.figure(figsize=figsize)
    sns.set_style("whitegrid")
    palette = sns.color_palette("husl", 2)
    
    # Extract data for plotting
    perturbations = [x['perturbation_size'] for x in robustness_data]
    models = ['Attention', 'Hyena']
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: L2 Distance vs Perturbation Size
    for i, model in enumerate(models):
        distances = [x[model]['l2_distance'] for x in robustness_data]
        ax1.plot(perturbations, distances, 
                marker='o', label=model, color=palette[i], linewidth=2.5)
    
    ax1.set_xlabel('Perturbation Size (ε)', fontsize=12)
    ax1.set_ylabel('L2 Distance from Original', fontsize=12)
    ax1.set_title('Modification Magnitude Required', fontsize=14)
    ax1.legend()
    
    # Plot 2: Success Rate vs Perturbation Size
    for i, model in enumerate(models):
        success_rates = [np.mean([x[model]['success'] for x in robustness_data[:j+1]]) 
                        for j in range(len(robustness_data))]
        ax2.plot(perturbations, success_rates, 
                marker='s', label=model, color=palette[i], linewidth=2.5)
    
    ax2.set_xlabel('Perturbation Size (ε)', fontsize=12)
    ax2.set_ylabel('Success Rate (%)', fontsize=12)
    ax2.set_title('Counterfactual Success Rate', fontsize=14)
    ax2.legend()
    
    # Add annotations for key observations
    def add_annotations(ax):
        for x, data in zip(perturbations, robustness_data):
            for i, model in enumerate(models):
                ax.annotate(f"{data[model]['confidence']:.2f}", 
                           (x, data[model]['l2_distance'] if ax == ax1 else data[model]['success']),
                           textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    add_annotations(ax1)
    add_annotations(ax2)
    
    plt.suptitle(title, fontsize=16, y=1.05)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

    # Additional statistical analysis
    print("\nStatistical Summary:")
    for model in models:
        avg_dist = np.mean([x[model]['l2_distance'] for x in robustness_data])
        success_rate = np.mean([x[model]['success'] for x in robustness_data])
        print(f"{model}: Avg L2 Distance = {avg_dist:.4f}, Success Rate = {success_rate:.2%}")

def class_transformation_experiment(model, image_path, original_class, target_classes):
    """
    Compare how much image modification is needed to change classification
    from original_class to various target_classes
    """
    transform = VAL_TRANSFORMATION
    original_image = Image.open(image_path).convert("RGB")
    original_image = transform(original_image).unsqueeze(0).to(DEVICE)
    results = {}
    for target in target_classes:
        print(f"\nTransforming {original_class} → {target}")
        
        # Generate counterfactual
            # Generate counterfactual - now returns a dictionary
        cf_result = counterfactual(
            model, 
            image_path,
            target_class=target,
            alpha=4.0,  # Stronger prediction loss
            beta=0.002,  # Weaker proximity constraint
            num_steps=100
        )
        
        # Store metrics - access values from dictionary
        results[target] = {
            'l2_distance': cf_result['metrics']['l2_distance'],
            'confidence': cf_result['metrics']['final_confidence'],
            'steps_to_converge': cf_result['metrics'].get('convergence_step', 300)  # Default if not present
        }
    
    return results

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
    #show_image(original_image, "Original Image")
    #show_image(input_image, "Counterfactual Image")
    #show_difference(original_image, input_image)

     # Histogram analysis
    #plot_histograms(original_image, input_image, title="Histogram Analysis: Original vs Counterfactual")
    # plot_difference_histogram(original_image, input_image, title="Histogram of Pixel Differences")

    difference_map = denormalize(input_image) - denormalize(original_image)
    max_diff = max(abs(difference_map.min()), abs(difference_map.max()))
    norm_diff = difference_map / max_diff

    return {
        "counterfactual_image": input_image.detach().cpu(),
        "original_image": original_image.detach().cpu(),
        "metrics": {
            "final_confidence": confidence,
            "l2_distance": proximity.item(),
            "min_perturbation": torch.norm(difference_map).item(),
            "success": confidence > 0.5,
            "loss_history": {
                "total_loss": loss.item(),
                "prediction_loss": target_score.item(),
                "proximity_loss": proximity.item()
            }
        },
        "visualizations": {
            "difference_map": difference_map.cpu(),
            "normalized_difference": norm_diff.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        }
}


def compute_spatial_frequencies(difference_map):
    """Analyze modification patterns in frequency domain"""
    # 2D FFT of difference map (expects H x W x C)
    fft = fftpack.fft2(difference_map, axes=(0,1))
    magnitude = np.abs(fft)
    
    # Create frequency bands
    h, w = magnitude.shape[:2]
    y_freq = np.fft.fftfreq(h)[:, np.newaxis]
    x_freq = np.fft.fftfreq(w)[np.newaxis, :]
    radial_freq = np.sqrt(x_freq**2 + y_freq**2)
    
    # Bin frequencies
    low_freq = magnitude[radial_freq < 0.1].mean()
    mid_freq = magnitude[(radial_freq >= 0.1) & (radial_freq < 0.3)].mean()
    high_freq = magnitude[radial_freq >= 0.3].mean()
    
    return {
        'low_freq_energy': low_freq,
        'mid_freq_energy': mid_freq,
        'high_freq_energy': high_freq,
        'fft_magnitude': magnitude
    }


def plot_frequency_analysis(attention_diff, hyena_diff):
    """Compare spatial modification patterns with full frequency domain visualization"""
    fig = plt.figure(figsize=(12, 5)) 

        
    # Frequency domain visualization
    def plot_fft(ax, diff_map, title):
        # Take single channel and compute FFT
        channel = diff_map[..., 0]  # Use first channel
        fft = fftpack.fft2(channel)
        fft_shifted = fftpack.fftshift(fft)
        magnitude = np.log(np.abs(fft_shifted) + 1e-8)  # Log scale for visibility
        
        ax.imshow(magnitude, cmap='viridis')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('Low Freq → High Freq')
    
    ax1 = fig.add_subplot(1, 3, 1)  # 1 row, 3 cols, position 1
    plot_fft(ax1, attention_diff, 'Attention Frequency Spectrum')
    
    ax2 = fig.add_subplot(1, 3, 2)  # Position 2
    plot_fft(ax2, hyena_diff, 'Hyena Frequency Spectrum')
    
    # Energy distribution comparison
    def compute_energy(diff_map):
        fft = fftpack.fft2(diff_map[..., 0])
        freq = np.fft.fftfreq(diff_map.shape[0])
        radial_freq = np.sqrt(np.fft.fftfreq(diff_map.shape[0])[:,None]**2 + 
                     np.fft.fftfreq(diff_map.shape[1])[None,:]**2)
        
        return {
            'low': np.mean(np.abs(fft)[radial_freq < 0.1]),
            'mid': np.mean(np.abs(fft)[(radial_freq >= 0.1) & (radial_freq < 0.3)]),
            'high': np.mean(np.abs(fft)[radial_freq >= 0.3])
        }
    
    attn_energy = compute_energy(attention_diff)
    hyena_energy = compute_energy(hyena_diff)
    
    ax3 = fig.add_subplot(1, 3, 3)  # Position 3
    x = np.arange(3)
    ax3.bar(x - 0.15, [attn_energy['low'], attn_energy['mid'], attn_energy['high']], 
            width=0.3, label='Attention')
    ax3.bar(x + 0.15, [hyena_energy['low'], hyena_energy['mid'], hyena_energy['high']], 
            width=0.3, label='Hyena')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Low', 'Mid', 'High'])
    ax3.set_ylabel('Energy (log)')
    ax3.set_title('Frequency Energy')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()



def compare_feature_importance(
    attention_model,
    hyena_model,
    image_path,
    original_class,
    target_classes,
    alpha=10.0,
    beta=0.01,
    num_steps=200
):
    """
    Compare feature importance via counterfactual difference maps between Attention and Hyena models.
    
    Args:
        attention_model: ViT with attention
        hyena_model: ViT with Hyena operators
        image_path: Path to input image
        original_class: Source class ID
        target_classes: List of target class IDs to test
        alpha: Prediction loss weight
        beta: Proximity loss weight
        num_steps: Optimization steps
    """
    # Load and preprocess image
    transform = VAL_TRANSFORMATION
    original_image = Image.open(image_path).convert("RGB")
    img_tensor = transform(original_image).unsqueeze(0).to(DEVICE)
    
    for target_class in target_classes:
        print(f"\nFeature Importance: {original_class} → {target_class}")
        
        # Generate counterfactuals
        attn_result = counterfactual(attention_model, image_path, target_class, alpha, beta, num_steps)
        hyena_result = counterfactual(hyena_model, image_path, target_class, alpha, beta, num_steps)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Feature Importance Comparison: {original_class} → {target_class}", fontsize=14)
        
        # Original image
        axes[0].imshow(denormalize(img_tensor).squeeze().permute(1, 2, 0).cpu().numpy())
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Attention difference map (normalized)
        attn_diff = attn_result["visualizations"]["normalized_difference"]
        im = axes[1].imshow(attn_diff, cmap='seismic', vmin=-1, vmax=1)
        axes[1].set_title(f"Attention Modifications\n(L2: {attn_result['metrics']['l2_distance']:.2f})")
        axes[1].axis('off')
        
        # Hyena difference map (normalized)
        hyena_diff = hyena_result["visualizations"]["normalized_difference"]
        axes[2].imshow(hyena_diff, cmap='seismic', vmin=-1, vmax=1)
        axes[2].set_title(f"Hyena Modifications\n(L2: {hyena_result['metrics']['l2_distance']:.2f})")
        axes[2].axis('off')
        
        # Add colorbar
       # fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', 
        #       fraction=0.05, pad=0.04, label='Modification Intensity')
        
        plt.tight_layout()
        plt.show()
        
        # Print localization metrics
        def get_localization_metrics(diff_map):
            """Quantify how localized changes are"""
            abs_diff = np.abs(diff_map).mean(axis=-1)
            threshold = 0.3 * abs_diff.max()
            binary_mask = abs_diff > threshold
            n_patches = len(np.unique(label(binary_mask))) - 1
            return n_patches, binary_mask.mean()
        
        attn_patches, attn_coverage = get_localization_metrics(attn_diff)
        hyena_patches, hyena_coverage = get_localization_metrics(hyena_diff)
        
        print("\nLocalization Metrics:")
        print(f"Attention: {attn_patches} distinct patches, {attn_coverage:.1%} image coverage")
        print(f"Hyena:     {hyena_patches} distinct patches, {hyena_coverage:.1%} image coverage")













from sklearn.manifold import TSNE
from scipy.spatial import distance_matrix

def decision_boundary_analysis(
    attention_model,
    hyena_model,
    image_path,
    original_class,
    target_classes,
    alpha=10.0,
    beta=0.01,
    num_steps=200
):
    """
    Compare decision boundaries by generating counterfactuals for multiple targets.
    
    Args:
        attention_model: ViT with standard attention
        hyena_model: ViT with Hyena operators
        image_path: Path to input image
        original_class: Ground truth class ID
        target_classes: List of target class IDs to test
        alpha: Prediction loss weight
        beta: Proximity loss weight
        num_steps: Optimization steps
    
    Returns:
        Dict containing metrics and visualization data
    """
    # Initialize results storage
    results = {
        'attention': {'l2_distances': [], 'diff_maps': []},
        'hyena': {'l2_distances': [], 'diff_maps': []},
        'target_classes': target_classes
    }
    
    # Generate counterfactuals for both models
    for target in target_classes:
        print(f"\nGenerating {original_class} → {target}")
        
        # Attention model
        attn_cf = counterfactual(
            attention_model, image_path, target, alpha, beta, num_steps
        )
        results['attention']['l2_distances'].append(attn_cf['metrics']['l2_distance'])
        results['attention']['diff_maps'].append(attn_cf['visualizations']['normalized_difference'])
        
        # Hyena model
        hyena_cf = counterfactual(
            hyena_model, image_path, target, alpha, beta, num_steps
        )
        results['hyena']['l2_distances'].append(hyena_cf['metrics']['l2_distance'])
        results['hyena']['diff_maps'].append(hyena_cf['visualizations']['normalized_difference'])
    
    # Visualization 1: Perturbation Distance Comparison
    plt.figure(figsize=(10, 5))
    x = np.arange(len(target_classes))
    width = 0.35
    
    plt.bar(x - width/2, results['attention']['l2_distances'], width, label='Attention')
    plt.bar(x + width/2, results['hyena']['l2_distances'], width, label='Hyena')
    
    plt.xlabel('Target Class')
    plt.ylabel('L2 Perturbation Distance')
    plt.title('Minimum Perturbation Required for Class Change')
    plt.xticks(x, target_classes)
    plt.legend()
    plt.show()
    
    # Visualization 2: Modification Pattern Comparison
    fig, axes = plt.subplots(2, len(target_classes), figsize=(15, 6))
    fig.suptitle('Difference Maps Across Target Classes', y=1.05)
    
    for i, target in enumerate(target_classes):
        # Attention difference maps
        axes[0,i].imshow(results['attention']['diff_maps'][i], cmap='seismic', vmin=-1, vmax=1)
        axes[0,i].set_title(f'Attn→{target}')
        axes[0,i].axis('off')
        
        # Hyena difference maps
        axes[1,i].imshow(results['hyena']['diff_maps'][i], cmap='seismic', vmin=-1, vmax=1)
        axes[1,i].set_title(f'Hyena→{target}')
        axes[1,i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Visualization 3: Decision Boundary Geometry Analysis
    all_diffs = np.array(results['attention']['diff_maps'] + results['hyena']['diff_maps'])
    flattened_diffs = all_diffs.reshape(len(target_classes)*2, -1)
    
    # t-SNE projection
    tsne = TSNE(n_components=2, perplexity=min(5, len(target_classes)-1))
    embeddings = tsne.fit_transform(flattened_diffs)
    
    plt.figure(figsize=(8, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(target_classes)))
    
    for i, target in enumerate(target_classes):
        # Plot attention and hyena points for each target
        plt.scatter(embeddings[i,0], embeddings[i,1], color=colors[i], 
                   marker='o', label=f'Attn→{target}')
        plt.scatter(embeddings[i+len(target_classes),0], 
                   embeddings[i+len(target_classes),1], 
                   color=colors[i], marker='s', label=f'Hyena→{target}')
    
    plt.title('t-SNE of Modification Patterns')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.show()
    
    # Calculate modification consistency
    def calculate_consistency(diff_maps):
        """Measure how similar modifications are across targets"""
        pairwise_dists = distance_matrix(diff_maps, diff_maps)
        return pairwise_dists.mean()
    
    attn_consistency = calculate_consistency(
        np.array(results['attention']['diff_maps']).reshape(len(target_classes), -1)
    )
    hyena_consistency = calculate_consistency(
        np.array(results['hyena']['diff_maps']).reshape(len(target_classes), -1)
    )
    
    print(f"\nDecision Boundary Characteristics:")
    print(f"Attention - Average pairwise modification distance: {attn_consistency:.2f}")
    print(f"Hyena - Average pairwise modification distance: {hyena_consistency:.2f}")
    
    return results