import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
from .Visualization import preprocess_image
from util import IMAGE_SIZE, MEAN, STD
from dataloaders.dataset.caltech256 import Caltech256Dataset


def calculate_Lime(model, image_path, device, dataset, model_type):

    def predict(images):
        normalize = transforms.Normalize(mean=MEAN, std=STD)

        model.eval()
        images = torch.stack([normalize(transforms.ToTensor()(img)) for img in images]).to(device)
        images = torch.nn.functional.interpolate(images, size=(IMAGE_SIZE, IMAGE_SIZE))
        output = model(images)
        probabilities = torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy()
        return probabilities

    model.eval()
    input_tensor = preprocess_image(image_path).to(device)

     # Forward pass to get the predicted class
    output = model(input_tensor)

    _, target_class = torch.max(output, 1)
    target_class = target_class.item()
    print(target_class)
    # Load the image for visualization
    img = Image.open(image_path)
    img = np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE)))
    
    # Initialize LIME
    explainer = lime_image.LimeImageExplainer()
    

    # Explain the prediction using LIME
    explanation = explainer.explain_instance(
        img, 
        classifier_fn=predict, 
        top_labels=5, 
        hide_color=0, 
        num_samples=1000
    )
    print(f"Available labels in explanation: {explanation.top_labels}")
    #print(target_class)
    # Get the image and mask for the top class
    temp, mask = explanation.get_image_and_mask(
        target_class, 
        positive_only=True, 
        num_features=10, 
        hide_rest=False
    )

    # Show the result
    plt.imshow(mark_boundaries(temp, mask,mode='inner'))
    plt.axis('off')
    plt.savefig(f"./{model_type}_lime.png", bbox_inches='tight', pad_inches=0)
    
    

if __name__ == "__main__":
    # Load a pretrained model
    model = models.resnet50(pretrained=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Preprocess the input image
    img_path = 'D:/projects/thesis/hyenaThesis/data/256_ObjectCategories/001.ak47/001_0002.jpg'  # Replace with the path to your image

    calculate_Lime(model,img_path,device)
    