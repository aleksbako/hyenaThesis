import argparse
import cv2
import numpy as np
import torch
import timm
import os
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args

def apply_perturbation(image, perturbation_type="noise", intensity=0.05):
    """
    Apply a small perturbation to the image.
    :param image: The input image as a NumPy array.
    :param perturbation_type: The type of perturbation ('noise', 'blur', 'shift').
    :param intensity: Intensity of the perturbation.
    :return: Perturbed image.
    """
    if perturbation_type == "noise":
        noise = np.random.normal(0, intensity, image.shape)  # Add Gaussian noise
        perturbed_image = np.clip(image + noise, 0, 1)
    elif perturbation_type == "blur":
        perturbed_image = cv2.GaussianBlur(image, (5, 5), intensity)  # Apply Gaussian blur
    elif perturbation_type == "shift":
        rows, cols, _ = image.shape
        M = np.float32([[1, 0, intensity * cols], [0, 1, intensity * rows]])  # Shift the image
        perturbed_image = cv2.warpAffine(image, M, (cols, rows))
    else:
        perturbed_image = image  # No perturbation if an unknown type is passed
    return perturbed_image

def reshape_transform(tensor, height=14, width=14):
    #print(f"Original tensor shape: {tensor.shape}")
    if isinstance(tensor, tuple):
        tensor = tensor[0]
    result = tensor[:, 1:, :].reshape(tensor.size(0),
                                      height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def calculate_vit_grad_cam(model,target_layers, img_path, model_type, layer_name, isPerturbed=False,perturbation_type="noise", intensity=0.04):
    """ python vit_gradcam.py --image-path <path_to_image>
    Example usage of using cam-methods on a VIT network.

    """

    model.eval()

    cam = GradCAM(model=model,
                                   target_layers=target_layers,
                                   reshape_transform=reshape_transform)

    rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
    rgb_img = cv2.resize(rgb_img, (224, 224))
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    input_tensor.requires_grad = True  # Ensure gradients are enabled

    output = model(input_tensor.to('cuda'))

    # Get the predicted label
    _, orignal_predicted_label = torch.max(output, 1)
    print(orignal_predicted_label)
    if isPerturbed:
        rgb_img = apply_perturbation(rgb_img, perturbation_type, intensity)
        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).float()
        input_tensor.requires_grad = True  # Ensure gradients are enabled

        # Forward pass through the model to get the prediction
    output = model(input_tensor.to('cuda'))

    # Get the predicted label
    _, predicted_label = torch.max(output, 1)
    cam_batch_size = 32
    
    print(f"Predicted label: {predicted_label.item()}")
    output[:, predicted_label].backward(retain_graph=True)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    for idx, target_layer in enumerate(target_layers):
        print(f"Processing layer: {target_layer}")

        # Initialize Grad-CAM with the current target layer
        cam = GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform)
        cam.batch_size = cam_batch_size

        # Generate the CAM heatmap for the input image
        grayscale_cam = cam(input_tensor=input_tensor)

        # Extract the CAM for the current image (since we have a single image in the batch)
        grayscale_cam = grayscale_cam[0, :]

        # Overlay the heatmap on the original image
        cam_image = show_cam_on_image(rgb_img, grayscale_cam)

        # Save the CAM image with a filename based on the layer index and model type
        layer_name = str(target_layer).replace('/', '_')  # Handle layer name formatting
        #layer_name = str(idx).replace('/', '_')  # Handle layer name formatting
        
        output_filename = f'../output/grad_cam/{orignal_predicted_label.item()+1}/{model_type}_cam_layer_{idx}.jpg'
        if isPerturbed:
            output_filename = f'../output/grad_cam/{orignal_predicted_label.item()+1}/perturbation/{perturbation_type}/{model_type}_cam_layer_{idx}_perturbed_intensity_{intensity}_To_{predicted_label.item()+1}.jpg'
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        cv2.imwrite(output_filename, cam_image)

        print(f"Saved CAM image for layer {layer_name} as {output_filename}")
  