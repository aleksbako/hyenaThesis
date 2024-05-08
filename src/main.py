import torch
import torchvision
import torch.nn as nn
import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader , random_split
from datetime import  timedelta
from sklearn.metrics import accuracy_score
from HyenaOperator import HyenaOperator
from models.hyenaVit import HyenaVit
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.init as init
from models.ModifiedVit import ModifiedVit
from dataloaders.dataset.caltech256 import Caltech256Dataset
import time
from models.SimpleViT import SimpleViT
from models.SimpleHyenaViT import SimpleHyenaViT

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            init.constant_(module.bias, 0)

def train(model, dataloader, loss, optimizer,model_type, warmup_epochs=5, checkpoint_interval=1, epochs=32, start_epoch=0, epoch_times=[], mean_loss=[], mean_accuracy=[]):
    

    model.train()

    for epoch in range(start_epoch, epochs):
        current_losses = []
        correct_predictions = 0
        total_samples = 0
        start_time = time.time()


        for data in tqdm.tqdm(dataloader):
            optimizer.zero_grad()

            image = data[0].to("cuda")

            label = data[1].to("cuda")

            pred_label = model(image)

            current_loss = loss(pred_label, label)

            _, predicted = torch.max(pred_label, 1)
            correct_predictions += (predicted == label).sum().item()
            total_samples += label.size(0)

            current_loss.backward()
            optimizer.step()
            current_losses.append(np.mean(current_loss.item()))

        end_time = time.time()
        epoch_training_time = end_time - start_time
        epoch_times.append(epoch_training_time)
        mean_accuracy.append(correct_predictions / total_samples * 100)
        mean_loss.append(np.mean(current_losses))

        print(f"{epoch}. epoch completed during training process with avg losses : {np.mean(mean_loss)},avg accuracies: {np.mean(mean_accuracy)} Total Training Time: {str(timedelta(seconds=np.sum(epoch_times)))}")

       
        if epoch % checkpoint_interval == 0:
            torch.save({
                'epoch': epoch,
                'model': model,
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch_times': epoch_times,
                'mean_accuracy': mean_accuracy,
                'mean_loss': mean_loss,

            }, f"../output/{model_type}_checkpoint.pt")

    return model
        
def get_model(baseline_model, model_type, train_dataloader,loss, lr, weight_decay, epochs):
   
    try:
        checkpoint = torch.load(f"../output/{model_type}_checkpoint.pt")
        start_epoch = checkpoint['epoch']
        model = checkpoint['model'].to('cuda')
        optim = torch.optim.AdamW(model.parameters(), lr=lr)
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        epoch_times = checkpoint['epoch_times']
        mean_accuracy = checkpoint['mean_accuracy']
        mean_loss = checkpoint['mean_loss']

    except Exception as e:
        start_epoch = 0
        model = baseline_model
        optim = torch.optim.AdamW(model.parameters(), lr=lr)
        epoch_times = []
        mean_accuracy = []
        mean_loss = []

    if start_epoch < epochs:
        
        model = train(model,train_dataloader,loss,optim,model_type, epochs=epochs, start_epoch=start_epoch , epoch_times=epoch_times, mean_loss=mean_loss, mean_accuracy=mean_accuracy)
    
    return model

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
    

def validate(model, dataloader, loss):
    model.eval()
    total_samples = 0
    total_loss = 0
    correct_predictions = 0

    with torch.no_grad():
        for data in tqdm.tqdm(dataloader):
            image = data[0].to("cuda")
            label = data[1].to("cuda")

            pred_label = model(image)
            current_loss = loss(pred_label, label)
            total_loss += current_loss.item()

            _, predicted = torch.max(pred_label, 1)
            correct_predictions += (predicted == label).sum().item()

            total_samples += label.size(0)

    average_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples * 100

    print(f"Validation Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%")



if __name__ == "__main__":

    root_dir = "../data/256_ObjectCategories/"
    epochs = 32
    lr = (10**(-5))
    loss = nn.CrossEntropyLoss()

    
    batch_size = 64
    transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.Resize((224,224))
    ]
    )

    cal_dataset = Caltech256Dataset("../data/256_ObjectCategories/",transform)
    train_size = int(0.8 * len(cal_dataset))
    val_size = len(cal_dataset) - train_size

    train_dataset, val_dataset = random_split(cal_dataset, [train_size, val_size], torch.Generator().manual_seed(24))
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size)
    model_type = "model"
    weight_decay = 0.05
    
    baseline = torchvision.models.vit_b_16(pretrained=False).to("cuda")
    ViT = SimpleViT(image_size = 224,
    patch_size = 16,
    num_classes = 257,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048).to('cuda')
    ViT.apply(init_weights)
    


    model = get_model(ViT, model_type, train_dataloader,loss, lr, weight_decay, epochs)
    
    hyena_ViT =  SimpleHyenaViT(image_size = 224,
    patch_size = 16,
    num_classes = 257,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048).to('cuda')   
    hyena_ViT.apply(init_weights)
    hyenaLoss = nn.CrossEntropyLoss()
    hyenaLr=1*(10**(-5))
    heyena_weight_decay = 0.01

    model_type = "hyena"
    

    model_hyena = get_model(hyena_ViT, model_type, train_dataloader, hyenaLoss, hyenaLr, heyena_weight_decay, epochs)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    validate(model, val_dataloader, loss)
    validate(model_hyena, val_dataloader, hyenaLoss)
    plot_metrics( output_dir="../output/")


    
    

    



 
    