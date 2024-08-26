import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
from .Visualization import preprocess_image



def calculate_Lime(model, image_path, device):

    def predict(images):
        model.eval()
        images = torch.stack([transforms.ToTensor()(img) for img in images]).to(device)
        images = torch.nn.functional.interpolate(images, size=(224, 224))
        output = model(images)
        probabilities = torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy()
        return probabilities

    model.eval()
    input_tensor = preprocess_image(image_path).to(device)

     # Forward pass to get the predicted class
    output = model(input_tensor)
    target_class = output.argmax().item()

    # Load the image for visualization
    img = Image.open(image_path)
    img = np.array(img.resize((224, 224)))

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
    plt.show()
    
    

if __name__ == "__main__":
    # Load a pretrained model
    model = models.resnet50(pretrained=True)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Preprocess the input image
    img_path = 'D:/projects/thesis/hyenaThesis/data/256_ObjectCategories/001.ak47/001_0002.jpg'  # Replace with the path to your image

    calculate_Lime(model,img_path,device)
    