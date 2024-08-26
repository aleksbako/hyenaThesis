import torch
import torchvision
import torch.nn as nn
import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader , random_split
from datetime import  timedelta
from sklearn.metrics import accuracy_score, precision_score
from HyenaOperator import HyenaOperator
from models.hyenaVit import HyenaVit
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.init as init
from models.ModifiedVit import ModifiedVit
from dataloaders.dataset.caltech256 import Caltech256Dataset
import time
from models.SimpleViT import SimpleViT
#from models.SimpleHyenaViT import SimpleHyenaViT
from models.SimpleHyenaViT import SimpleHyenaViT
from sklearn.metrics import classification_report
from util.Lime import calculate_Lime
from util.Visualization import plot_metrics
from util.GradCAM import calculate_Grad_CAM
from util.GradCAMViT import calculate_vit_grad_cam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            init.constant_(module.bias, 0)

def train(model, train_dataloader, val_dataloader, loss, optimizer,model_type,scheduler, warmup_epochs=5, checkpoint_interval=1, epochs=32, start_epoch=0, epoch_times=[], mean_loss=[], mean_accuracy=[],best_accuracy=0, best_precision=0):
    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    train_losses = []
    val_losses = []
    
    for epoch in range(start_epoch, epochs):
        current_losses = []
        correct_predictions = 0
        total_samples = 0
        start_time = time.time()
        model.train()

        for data in tqdm.tqdm(train_dataloader):
            optimizer.zero_grad()
            

            image = data[0].to("cuda")

            label = data[1].to("cuda")
            
            pred_label = model(image)

            current_loss = loss(pred_label, label)
            current_loss.backward()
            optimizer.step()

            _, predicted = torch.max(pred_label, 1)
            correct_predictions += (predicted == label).sum().item()
            total_samples += label.size(0)

            current_losses.append(np.mean(current_loss.item()))

        end_time = time.time()
        epoch_training_time = end_time - start_time
        epoch_times.append(epoch_training_time)
        current_accuracy = correct_predictions / total_samples * 100
        mean_accuracy.append(current_accuracy)
        mean_loss.append(np.mean(current_losses))
        train_losses.append(np.mean(current_losses))
        print(f"{epoch}. epoch completed during training process with current epoch loss : {np.mean(current_losses)}, accuracies: {current_accuracy} Total Training Time: {str(timedelta(seconds=np.sum(epoch_times)))}")

          # Validate the model after each epoch
        val_loss, val_accuracy, val_precision = validate(model, val_dataloader, loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        # Update the best model based on validation accuracy and precision
        if val_accuracy > best_accuracy or val_precision > best_precision:
            best_accuracy = val_accuracy
            best_precision = val_precision
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch_times': epoch_times,
            'mean_accuracy': mean_accuracy,
            'mean_loss': mean_loss,
            'best_accuracy': best_accuracy,
            'best_precision': best_precision,
            }, f"../output/{model_type}_checkpoint.pt")
        plt.figure(figsize=(10, 6))
        plt.plot(range(start_epoch, epoch+1), train_losses, label='Training Loss')
        plt.plot(range(start_epoch, epoch+1), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'{model_type} Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"../output/{model_type}_loss_curve.png")
      
            
      

    return model
        
def get_model(baseline_model, model_type, train_dataloader,val_dataloader, loss, lr, weight_decay, epochs):
    best_accuracy = best_precision = 0
    try:
        print(f"{model_type}_checkpoint.pt")
        checkpoint = torch.load(f"../output/{model_type}_checkpoint.pt")
        
          # Load the state dictionary into the baseline_model
        baseline_model.load_state_dict(checkpoint['model'])
    
        
        # Now model refers to the baseline_model with loaded weights
        model = baseline_model
    
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        epoch_times = checkpoint['epoch_times']
        mean_accuracy = checkpoint['mean_accuracy']
        mean_loss = checkpoint['mean_loss']
        best_accuracy = checkpoint['best_accuracy']
        best_precision = checkpoint['best_precision']
        

    except Exception as e:
        print(e)
        start_epoch = 0
        model = baseline_model
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,betas=[0.9,0.999])
        epoch_times = []
        mean_accuracy = []
        mean_loss = []
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=32)
    scheduler = ReduceLROnPlateau(optim, mode='min', patience=3, verbose=True)
    print(start_epoch)
    if start_epoch < epochs:
        
        model = train(model,train_dataloader,val_dataloader,loss,optim,model_type, scheduler,epochs=epochs, start_epoch=start_epoch , epoch_times=epoch_times, mean_loss=mean_loss, mean_accuracy=mean_accuracy , best_accuracy=best_accuracy, best_precision=best_precision)
    
    return model



def GetMedianTime(output_dir="../output/"):
    
    try:

        model1_checkpoint = torch.load(f"../output/model_checkpoint.pt")
        
        model2_checkpoint = torch.load(f"../output/hyena_checkpoint.pt")

        print(f"Min time spent for ViT model : {np.min(model1_checkpoint['epoch_times'])}")
        print(f"Min time spent for Hyena ViT model : {np.min(model2_checkpoint['epoch_times'])}")
        # Adjust epoch times to be cumulative
        print(f"Median time spent for ViT model : {np.median(model1_checkpoint['epoch_times'])}")
        print(f"Median time spent for Hyena ViT model : {np.median(model2_checkpoint['epoch_times'])}")
      #  model2_epoch_times_cumulative = [sum(model2_checkpoint['epoch_times'][:i+1]) for i in range(len(model2_checkpoint['epoch_times']))]
        print(f"Max time spent for ViT model : {np.max(model1_checkpoint['epoch_times'])}")
        print(f"Max time spent for Hyena ViT model : {np.max(model2_checkpoint['epoch_times'])}")
    except:
        print("error when fetching  epoch time data")
    

    

def validate(model, dataloader, loss):
    model.eval()
    total_samples = 0
    total_loss = 0
    correct_predictions = 0
    all_predicted = []  # Accumulate predictions for the entire epoch
    all_labels = []     # Accumulate ground truth labels for the entire epoch

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
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
 
        
    average_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples * 100
    precision = precision_score(all_labels, all_predicted, average='weighted', zero_division=0.0) * 100
    
    #report = classification_report(all_labels, all_predicted)
   # print(report)

    print(f"Validation Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}%")
    return average_loss , accuracy, precision


import os

from sklearn.model_selection import ParameterGrid

def cross_validate(train_dataset, val_dataset, param_grid, loss_function, epochs, batch_size=64):
    best_model = None
    best_score = 0
    best_params = None
    results = []
    output_dir = "../output/"
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, "cross_validation_results.txt")
    # Create DataLoaders
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
 
 # Open the log file for writing
    with open(log_file_path, "a") as log_file:
        log_file.write("Cross-Validation Results\n")
        log_file.write("========================\n\n")
        log_file.flush()
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

        for params in ParameterGrid(param_grid):
            try:
                log_file.write(f"Training with parameters: {params}\n")
                log_file.flush()
                print(f"Training with parameters: {params}")

                train_transform = transforms.Compose([
                        transforms.Resize((params['image_size']*2, params['image_size']*2)),
                transforms.RandomCrop(params['image_size']),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
                    
                ]
                )

                val_transform = transforms.Compose([
                transforms.Resize((params['image_size'],params['image_size'])),                 # Resize to the same size as training set
                transforms.ToTensor(),                         # Convert images to tensor
                transforms.Normalize(mean=mean, std=std)       # Normalize with the same mean and std
            ])
                train_dataloader.dataset.dataset.transform = train_transform
                val_dataloader.dataset.dataset.transform = val_transform

                            
                
                # Instantiate the model with the current set of parameters
                model = SimpleHyenaViT(
                    image_size=params['image_size'],
                    patch_size=params['patch_size'],
                    num_classes=params['num_classes'],
                    dim=params['dim'],
                    depth=params['depth'],
                    heads=params['heads'],
                    mlp_dim=params['mlp_dim'],
                    l_max=params['l_max'],
                    filter_order=params['filter_order'],
                    dropout=params['dropout'],
                    filter_dropout=params['filter_dropout']
                ).to('cuda')

                optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, verbose=True)

                # Train the model
                trained_model = train(model, train_dataloader, val_dataloader, loss_function, optimizer, "hyenaVit", scheduler, epochs=epochs)

                # Validate the model
                val_loss, val_accuracy, val_precision = validate(trained_model, val_dataloader, loss_function)
                
                # Store results
                results.append({
                    'params': params,
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'val_precision': val_precision
                }) 

                log_file.write(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%, Precision: {val_precision:.2f}%\n\n")
                log_file.write("========================\n\n")
                log_file.flush()
                # Check if this is the best model so far
                if val_accuracy > best_score:
                    best_score = val_accuracy
                    best_model = trained_model
                    best_params = params

                print(f"Completed training with val_loss: {val_loss:.4f}, val_accuracy: {val_accuracy:.2f}%, val_precision: {val_precision:.2f}%\n")
            except Exception as e:
                log_file.write(f"An error occurred during training: {str(e)}\n")
                log_file.flush()
                print(f"An error occurred during training: {str(e)}")
                break  # Optionally, break the loop if an error occurs    

        log_file.write("Best parameters:\n")
        log_file.write(f"{best_params}\n")
        log_file.write(f"Best validation accuracy: {best_score:.2f}%\n")
        

    print(f"Best parameters: {best_params}")
    print(f"Best validation accuracy: {best_score:.2f}%")
    
    return best_model, best_params, results

if __name__ == "__main__":

    root_dir = "../data/256_ObjectCategories/"
    epochs = 33
    lr = 1e-4
    loss = nn.CrossEntropyLoss()



   
    batch_size = 100
    
    
    train_transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(45),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
         
    ]
    )

    val_transform = transforms.Compose([
    transforms.Resize((224,224)),                 # Resize to the same size as training set
    transforms.ToTensor(),
                         transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])                     # Convert images to tensor
         # Normalize with the same mean and std
])

    cal_dataset = Caltech256Dataset("../data/256_ObjectCategories/",transform=train_transform)
    #cifar_dataset = torchvision.datasets.CIFAR100("../data/",train=True, transform=None)

    train_size = int(0.6 * len(cal_dataset))
    val_size = len(cal_dataset) - train_size
    ##train_size = int(0.6 * len(cifar_dataset))
    #val_size = len(cifar_dataset) - train_size

    train_dataset, val_dataset = random_split(cal_dataset, [train_size, val_size], torch.Generator().manual_seed(42))
    train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # Apply the respective transformations
    #train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    model_type = "model"
    weight_decay =1e-3
   # ViT = torchvision.models.vit_b_16(pretrained=True)
    
    ViT = SimpleViT(image_size = 224,
    patch_size = 16,
    num_classes = 257,
     dim = 512,
    depth = 5,
    heads =12,
    mlp_dim = 1024,
    dropout=0.2,
    emb_dropout=0.3).to('cuda')
    ViT.apply(init_weights)
    

    
    model = get_model(ViT, model_type, train_dataloader, val_dataloader, loss, lr, weight_decay, epochs)
  
    #hyena_ViT =  SimpleHyenaViT(image_size = 224,
    #patch_size = 16,
    #num_classes = 257,
   # d_model = 512,
    #depth = 2,
  #  dropout = 0.3).to('cuda')   
    hyena_ViT =  SimpleHyenaViT(image_size = 224,
    patch_size = 16,
    num_classes = 257,
    dim = 512,
    
    depth = 6,
    heads = 1,
   mlp_dim = 512).to('cuda')

    hyena_ViT.apply(init_weights)
    hyenaLoss = nn.CrossEntropyLoss()
    hyenaLr=4e-6
    heyena_weight_decay = 1e-2
    
    model_type = "hyena"
    
    """   # Define parameter grid for cross-validation
    param_grid = {
        'image_size': [128,224],  # You can add other sizes if needed
        'patch_size': [16,32],
        'num_classes': [257],
        'dim': [512,786, 1024],
        'depth': [2,4,6, 7],
        'heads': [8],
        'mlp_dim': [1024],
        'l_max': [197],
        'filter_order':[16,32,48,64,128],
        'dropout': [0.2,0.5,0.6,0.7],
        'filter_dropout': [0.2,0.5,0.6,0.7],
        'lr': [1e-3,1e-4, 1e-5, 7e-5],
        'weight_decay': [5e-2, 1e-3, 5e-4]
    }
    """
    # Run cross-validation to find the best model and parameters
   # best_model, best_params, results = cross_validate(train_dataset, val_dataset, param_grid, hyenaLoss, 15, batch_size)
    

    model_hyena = get_model(hyena_ViT, model_type, train_dataloader,val_dataloader, hyenaLoss, hyenaLr, heyena_weight_decay, epochs)


    #validate(model_hyena, val_dataloader, hyenaLoss)
    validate(model, val_dataloader, loss)
    
    plot_metrics( output_dir="../output/")
    GetMedianTime()
    

    for batch in val_dataloader:
        images, labels = batch
        break  # Stop after the first batch

    
    image = images.to("cuda")
   # model.eval()
  
    img_path = 'D:/projects/thesis/hyenaThesis/data/256_ObjectCategories/001.ak47/001_0002.jpg'  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Hyena calculation
   # calculate_Lime(model_hyena, img_path, device)
   # target_layer = model.transformer.layers[-1][0] 
    #calculate_Grad_CAM(model,target_layer, img_path, device)

        
    #target_layer = [model_hyena.blocks[-1].attn]
    target_layer = [model_hyena.transformer.layers[-1][0]]

    calculate_vit_grad_cam(model_hyena, target_layer, img_path)
    #calculate_Lime(model, img_path, device)
    
   




    
    

    



 
    