import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision import transforms

def preprocess_image(img_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path)
    img = preprocess(img).unsqueeze(0)
    return img

def show_cam_on_image(img, mask):
    mask = np.uint8(255 * mask)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)



def plot_metrics(output_dir="../output/"):
    
    try:
        plt.figure(figsize=(15, 5))
        model1_checkpoint = torch.load(f"../output/model_checkpoint.pt")
        
        model2_checkpoint = torch.load(f"../output/hyena_checkpoint.pt")

        # Adjust epoch times to be cumulative
        model1_epoch_times_cumulative = [sum(model1_checkpoint['epoch_times'][:i+1]) for i in range(len(model1_checkpoint['epoch_times']))]
        model2_epoch_times_cumulative = [sum(model2_checkpoint['epoch_times'][:i+1]) for i in range(len(model2_checkpoint['epoch_times']))]

        # Plot Mean Accuracy
        plt.subplot(1, 3, 1)
        plt.plot(model1_checkpoint['mean_accuracy'], label='ViT')
        plt.plot(model2_checkpoint['mean_accuracy'], label='HyenaViT')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Accuracy')
        plt.title('Mean Accuracy Comparison')
        plt.legend()
        plt.savefig(output_dir + 'mean_accuracy_comparison.png')
        plt.close()

        # Plot Mean Loss
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(model1_checkpoint['mean_loss'], label='ViT')
        plt.plot(model2_checkpoint['mean_loss'], label='HyenaViT')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Loss')
        plt.title('Mean Loss Comparison')
        plt.legend()
        plt.savefig(output_dir + 'mean_loss_comparison.png')
        plt.close()

        # Plot Epoch Times
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(model1_epoch_times_cumulative, label='ViT')
        plt.plot(model2_epoch_times_cumulative, label='HyenaViT')
        plt.xlabel('Epochs')
        plt.ylabel('Cumulative Epoch Time (s)')
        plt.title('Cumulative Epoch Time Comparison')
        plt.legend()
        plt.savefig(output_dir + 'cumulative_epoch_time_comparison.png')
        plt.close()
    except:
        print("error when plotting")
