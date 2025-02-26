import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision import transforms
from util import IMAGE_SIZE, VAL_TRANSFORMATION
def preprocess_image(img_path):
    preprocess = VAL_TRANSFORMATION
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



def plot_metrics(model_type_1, model_type_2, output_dir="../output/"):
    
    try:
       
        model1_checkpoint = torch.load(f"../output/{model_type_1}_checkpoint.pt")
        
        model2_checkpoint = torch.load(f"../output/{model_type_2}_checkpoint.pt")

        # Adjust epoch times to be cumulative
        model1_epoch_times_cumulative = [sum(model1_checkpoint['epoch_times'][:i+1]) for i in range(len(model1_checkpoint['epoch_times']))]
        model2_epoch_times_cumulative = [sum(model2_checkpoint['epoch_times'][:i+1]) for i in range(len(model2_checkpoint['epoch_times']))]

        model1_epoch_val_times_cumulative = [sum(model1_checkpoint['epoch_val_times'][:i+1]) for i in range(len(model1_checkpoint['epoch_val_times']))]
        model2_epoch_val_times_cumulative = [sum(model2_checkpoint['epoch_val_times'][:i+1]) for i in range(len(model2_checkpoint['epoch_val_times']))]

        # Plot Mean Accuracy
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(model1_checkpoint['mean_train_accuracy'], label=model_type_1)
        plt.plot(model2_checkpoint['mean_train_accuracy'], label=model_type_2)
        plt.xlabel('Epochs')
        plt.ylabel('Mean Training Accuracy')
        plt.title('Mean Training Accuracy Comparison')
        plt.legend()
        plt.savefig(output_dir + 'mean_train_accuracy_comparison.png')
        plt.close()

         # Plot Mean Accuracy
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(model1_checkpoint['mean_val_accuracy'], label=model_type_1)
        plt.plot(model2_checkpoint['mean_val_accuracy'], label=model_type_2)
        plt.xlabel('Epochs')
        plt.ylabel('Mean Validation Accuracy')
        plt.title('Mean validation Accuracy Comparison')
        plt.legend()
        plt.savefig(output_dir + 'mean_val_accuracy_comparison.png')
        plt.close()


        # Plot Mean Loss
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(model1_checkpoint['mean_train_loss'], label=model_type_1)
        plt.plot(model2_checkpoint['mean_train_loss'], label=model_type_2)
        plt.xlabel('Epochs')
        plt.ylabel('Mean Training Loss')
        plt.title('Mean Training Loss Comparison')
        plt.legend()
        plt.savefig(output_dir + 'mean_train_loss_comparison.png')
        plt.close()

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(model1_checkpoint['mean_val_loss'], label=model_type_1)
        plt.plot(model2_checkpoint['mean_val_loss'], label=model_type_2)
        plt.xlabel('Epochs')
        plt.ylabel('Mean Validation Loss')
        plt.title('Mean Validation Loss Comparison')
        plt.legend()
        plt.savefig(output_dir + 'mean_val_loss_comparison.png')
        plt.close()

        # Plot Epoch Times
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(model1_epoch_times_cumulative, label=model_type_1)
        plt.plot(model2_epoch_times_cumulative, label=model_type_2)
        plt.xlabel('Epochs')
        plt.ylabel('Cumulative Training Epoch Time (s)')
        plt.title('Cumulative Training Epoch Time Comparison')
        plt.legend()
        plt.savefig(output_dir + 'cumulative_epoch_time_comparison.png')
        plt.close()

           # Plot Epoch Times
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(model1_epoch_val_times_cumulative, label=model_type_1)
        plt.plot(model2_epoch_val_times_cumulative, label=model_type_2)
        plt.xlabel('Epochs')
        plt.ylabel('Cumulative Validation Epoch Time (s)')
        plt.title('Cumulative Validation Epoch Time Comparison')
        plt.legend()
        plt.savefig(output_dir + 'cumulative_epoch_val_time_comparison.png')
        plt.close()
    except Exception as e:
        print(f"error when plotting : {e}")
