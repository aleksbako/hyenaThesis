# python function to get model output; replace this function with your own model function.
import shap
from .Visualization import preprocess_image
import cv2
import numpy as np
import torch
import torch.nn as nn
# Apply the replacement function
import torchvision.transforms as transforms
import torchvision
from PIL import Image
# Prepare data transformation pipeline
import torch.nn.functional as F
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)
    elif x.dim() == 3:
        x = x if x.shape[0] == 3 else x.permute(2, 0, 1)
    return x


def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[3] == 3 else x.permute(0, 2, 3, 1)
    elif x.dim() == 3:
        x = x if x.shape[2] == 3 else x.permute(1, 2, 0)
    return x


transform = [
    torchvision.transforms.Lambda(nhwc_to_nchw),
    torchvision.transforms.Lambda(lambda x: x * (1 / 255)),
    torchvision.transforms.Normalize(mean=mean, std=std),
    torchvision.transforms.Lambda(nchw_to_nhwc),
]

inv_transform = [
    torchvision.transforms.Lambda(nhwc_to_nchw),
    torchvision.transforms.Normalize(
        mean=(-1 * np.array(mean) / np.array(std)).tolist(),
        std=(1 / np.array(std)).tolist(),
    ),
    torchvision.transforms.Lambda(nchw_to_nhwc),
]

transform = torchvision.transforms.Compose(transform)
inv_transform = torchvision.transforms.Compose(inv_transform)



def replace_silu_with_non_inplace(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.SiLU):
            # Replace with non-inplace SiLU
            setattr(model, child_name, nn.SiLU(inplace=False))
        else:
            # Recursively apply to child modules
            replace_silu_with_non_inplace(child)

def blur_shap_values(shap_values, sigma=1):
    blurred_shap_values = []
    for shap_map in shap_values:
        # Ensure shap_map is 3D (height, width, channels)
        if shap_map.ndim == 3:
            blurred_map = cv2.GaussianBlur(shap_map, (5, 5), sigma)
            blurred_shap_values.append(blurred_map)
        else:
            # If the map is 2D, apply blur to each channel separately
            for channel in range(shap_map.shape[-1]):  # For each channel
                blurred_map = cv2.GaussianBlur(shap_map[..., channel], (5, 5), sigma)
                shap_map[..., channel] = blurred_map
            blurred_shap_values.append(shap_map)
    return blurred_shap_values

def get_top_predictions(model, loader, num_top=2):
    top_images = []
    top_scores = []
    
    # Ensure the model is in evaluation mode
    model.eval()
    
    with torch.no_grad():
        for images, _ in loader:
            images = images.to('cuda')
            outputs = model(images)
            
            # Assuming softmax for classification; adjust if using another output type
            probabilities = F.softmax(outputs, dim=1)
            confidence_scores, _ = torch.max(probabilities, dim=1)  # Highest score per image
            
            # Iterate through each image and score
            for i, score in enumerate(confidence_scores):
                if len(top_images) < num_top:
                    top_images.append(images[i])
                    top_scores.append(score)
                else:
                    # Find the lowest score in current top and replace if new score is higher
                    min_score_idx = torch.argmin(torch.tensor(top_scores))
                    if score > top_scores[min_score_idx]:
                        top_images[min_score_idx] = images[i]
                        top_scores[min_score_idx] = score

    # Stack selected images into a single tensor batch
    top_images = torch.stack(top_images).to('cuda')
    return top_images

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),  # Resize the image to 256x256
    #transforms.CenterCrop((IMAGE_SIZE,IMAGE_SIZE)),  # Crop the center of the image to 224x224                # Resize to the same size as training set
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)                   

])
def threshold_shap_values(shap_values, threshold=0.05):
    thresholded_shap_values = []
    for shap_map in shap_values:
        shap_map = np.abs(shap_map)  # Take absolute value to ignore negative values
        shap_map[shap_map < threshold] = 0  # Apply threshold
        thresholded_shap_values.append(shap_map)
    return thresholded_shap_values



def threshold_and_blur_shap_values(shap_values, threshold=0.05, sigma=1):
    """
    Apply thresholding and Gaussian blur to SHAP values.

    Parameters:
    - shap_values: List of SHAP value maps (height, width, channels)
    - threshold: Value below which SHAP values are set to 0
    - sigma: Standard deviation for the Gaussian blur (controls the level of smoothing)
    
    Returns:
    - blurred_shap_values: List of processed SHAP values after thresholding and blurring
    """
    blurred_shap_values = []
    
    for shap_map in shap_values:
        # Apply thresholding to remove weak SHAP values
        shap_map = np.abs(shap_map)  # Consider absolute value for thresholding
        
        # Apply the threshold: values below the threshold are set to 0
        shap_map[shap_map < threshold] = 0

        # Apply Gaussian blur to smooth the SHAP values
        if shap_map.ndim == 3:  # For color images (height, width, channels)
            blurred_map = cv2.GaussianBlur(shap_map, (5, 5), sigma)
            blurred_shap_values.append(blurred_map)
        else:  # For grayscale images
            blurred_map = cv2.GaussianBlur(shap_map, (5, 5), sigma)
            blurred_shap_values.append(blurred_map)

    return blurred_shap_values

def load_and_transform_image(image_path):
    # Load the image
    image = Image.open(image_path).convert("RGB")  # Ensure it’s in RGB mode
    
    # Apply transformations
    transformed_image = val_transform(image)
    
    # Add a batch dimension (since the model expects a batch, even if it’s just 1 image)
    transformed_image = transformed_image.unsqueeze(0)
    
    return transformed_image

def caluclateShap(model,loader, class_names, layers_to_show,images):
    def remove_inplace_silu(model):
        for module in model.modules():
            if isinstance(module, nn.SiLU):
                module.inplace = False

    remove_inplace_silu(model)
    
    image_list = [load_and_transform_image(path) for path in images]
    top_images = torch.cat(image_list, dim=0).to('cuda')
    # Select a set of background examples to take an expectation over
    background = next(iter(loader))[0][:100].to('cuda')  # Adjust size if needed
    
    # Initialize the SHAP explainer
    print("Initializing GradientExplainer...")
    e = shap.GradientExplainer(model, background)
    
    # Get SHAP values for a few images
    shap_values = e.shap_values(top_images)
    
    # Transpose for visualization compatibility
    sv_transposed = [shap_value.transpose(0, 2, 3, 1) for shap_value in shap_values]

    # Ensure images are transformed back correctly without negation
    input_images = top_images.permute(0, 2, 3, 1).cpu().numpy()
    input_images = np.clip(input_images, 0, 1)  # Optional: ensure values are within [0, 1]

    # Plot the SHAP explanations
    print("Plotting SHAP explanations...")
    #thresholded_shap_values = threshold_shap_values(sv_transposed, threshold=0.05)
    #blurred_shap_values = blur_shap_values(thresholded_shap_values, sigma=1)
    
    #shap.image_plot(blurred_shap_values, input_images)
    #thresholded_shap_values = threshold_shap_values(sv_transposed, threshold=0.05)
    shap.image_plot(sv_transposed, input_images)