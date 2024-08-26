import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models, transforms
from .Visualization import preprocess_image , show_cam_on_image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, target_class):
        # Forward pass
        output = self.model(input_tensor)
        target = output[:, target_class]

        # Backward pass
        self.model.zero_grad()
        target.backward(retain_graph=True)

         # Get gradients and activations
        gradients = self.gradients.data.cpu().numpy()
        activations = self.activations.data.cpu().numpy()


        print(f"Gradients shape: {gradients.shape}")
        print(f"Activations shape: {activations.shape}")
        # Check the shape of gradients and activations
        if gradients.ndim == 4:
            # Pool the gradients across the spatial dimensions
            weights = np.mean(gradients, axis=(2, 3))
            # Compute the weighted sum of the activations
            grad_cam = np.zeros(activations.shape[2:], dtype=np.float32)
            for i, w in enumerate(weights[0]):
                grad_cam += w * activations[0, i, :, :]
        else:
            # Pool the gradients across the sequence dimension
            weights = np.mean(gradients, axis=1)
            # Compute the weighted sum of the activations
            grad_cam = np.zeros(activations.shape[1:], dtype=np.float32)
            for i, w in enumerate(weights[0]):
                grad_cam += w * activations[0, i, :]

        # Apply ReLU to the weighted sum
        grad_cam = np.maximum(grad_cam, 0)

        # Normalize the Grad-CAM
        grad_cam = cv2.resize(grad_cam, (input_tensor.size(2), input_tensor.size(3)))
        grad_cam = grad_cam - np.min(grad_cam)
        grad_cam = grad_cam / np.max(grad_cam)
        return grad_cam



def calculate_Grad_CAM(model,target_layer, image_path, device):
    model.eval()


    # Initialize Grad-CAM
    grad_cam = GradCAM(model, target_layer)

    # Preprocess the input image
    input_tensor = preprocess_image(image_path).to(device)

    # Forward pass to get the predicted class
    output = model(input_tensor)
    target_class = output.argmax().item()

    # Generate Grad-CAM
    mask = grad_cam(input_tensor, target_class)

    # Load the image for visualization
    img = Image.open(img_path)
    img = np.array(img.resize((224, 224))) / 255.0

    # Visualize the result
    cam_image = show_cam_on_image(img, mask)
    plt.imshow(cam_image)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Load a pretrained model
    model = models.resnet50(pretrained=True)
    model.eval()

    # Specify the target layer (e.g., the last convolutional layer of ResNet50)
    target_layer = model.layer4[2].conv3


    # Preprocess the input image
    img_path = 'D:/projects/thesis/hyenaThesis/data/256_ObjectCategories/001.ak47/001_0002.jpg'  # Replace with the path to your image

    calculate_Grad_CAM(model,target_layer,img_path,'cuda')
  
