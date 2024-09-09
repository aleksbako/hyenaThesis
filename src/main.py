import torch
import torchvision
import torch.nn as nn
import tqdm
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader , random_split
from datetime import  timedelta
from sklearn.metrics import accuracy_score, precision_score
from models.hyenaVit import HyenaVit
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.init as init
from models.ModifiedVit import ModifiedVit
from dataloaders.dataset.caltech256 import Caltech256Dataset
from dataloaders.dataset.ImageNetValDataset import ImageNetValDataset
from models.SimpleViT import SimpleViT
from models.hyenaVit import HyenaVit
from models.SimpleHyenaViT import SimpleHyenaViT
from sklearn.metrics import classification_report
from util.Lime import calculate_Lime
from util.Visualization import plot_metrics
from util.GradCAM import calculate_Grad_CAM
from util.GradCAMViT import calculate_vit_grad_cam
from torch.optim.lr_scheduler import CosineAnnealingLR
from util import validate, train, OUTPUT_DIR, DATA_DIR, DEVICE
from util import IMAGE_SIZE, BATCH_SIZE, LOSS, EPOCH, MEAN, STD
from util import LEARNING_RATE, WEIGHT_DECAY
from util import HYENA_LEARNING_RATE, HYENA_WEIGHT_DECAY, HYENA_LOSS


import os

from sklearn.model_selection import ParameterGrid

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        init.constant_(module.weight, 1)
        init.constant_(module.bias, 0)
    elif isinstance(module, nn.Embedding):
        init.normal_(module.weight, mean=0, std=0.02)


        
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
    scheduler = CosineAnnealingLR(optim, T_max=10)
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
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
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

    root_dir = DATA_DIR + "256_ObjectCategories/"

    train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2,0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
   # transforms.RandomErasing(p=0.5),
   # transforms.GaussianBlur(kernel_size=5)
    ]
    )

    val_transform = transforms.Compose([
    transforms.Resize(256),  # Resize the image to 256x256
    transforms.CenterCrop(IMAGE_SIZE),  # Crop the center of the image to 224x224                # Resize to the same size as training set
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)                   

])

    cal_dataset = Caltech256Dataset("../data/256_ObjectCategories/",transform=train_transform)

    train_size = int(0.8 * len(cal_dataset))
    val_size = len(cal_dataset) - train_size

    train_dataset, val_dataset = random_split(cal_dataset, [train_size, val_size], torch.Generator().manual_seed(42))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Apply the respective transformations
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

   # ViT = SimpleViT(image_size = 224,
   # patch_size = 16,
  #  num_classes = 1001,
   #  dim = 512,
    #depth = 8,
   # heads =12,
   # mlp_dim = 2048,
   # dropout=0.1,
   # emb_dropout=0.1).to('cuda')
   # ViT.apply(init_weights)

    ViT = torchvision.models.vit_b_16(pretrained=True)
    model = get_model(ViT, "ViT", train_loader, val_loader, LOSS, LEARNING_RATE, WEIGHT_DECAY, EPOCH)
  
    #hyena_ViT =  SimpleHyenaViT(image_size = 224,
    #patch_size = 16,
    #num_classes = 257,
   # d_model = 512,
    #depth = 2,
  #  dropout = 0.3).to('cuda')   
    #hyena_ViT =  SimpleHyenaViT(image_size = 224,
   # patch_size = 16,
    #num_classes = 257,
   # dim = 512,
  # depth = 6,
   # heads = 1,
  # mlp_dim = 512).to('cuda')

   # hyena_ViT.apply(init_weights)
    
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
    #best_model, best_params, results = cross_validate(train_dataset, val_dataset, param_grid, hyenaLoss, 15, batch_size)
    
    hyena_ViT = HyenaVit().to('cuda')
    model_hyena = get_model(hyena_ViT, "ViT_with_Hyena", train_loader,val_loader, HYENA_LOSS, HYENA_LEARNING_RATE, HYENA_WEIGHT_DECAY, EPOCH)

    validate(model, val_loader, LOSS)
    validate(model_hyena, val_loader, HYENA_LOSS)
 #   
    
    plot_metrics( output_dir=OUTPUT_DIR)
    GetMedianTime()
    
  
    img_path = 'D:/projects/thesis/hyenaThesis/data/256_ObjectCategories/001.ak47/001_0002.jpg'  

    target_layer = [ViT.encoder.layers[-1].self_attention]

    calculate_vit_grad_cam(ViT, target_layer, img_path, "ViT" ,"final_attention_layer")

    calculate_Lime(ViT, img_path, DEVICE)
    
   




    
    

    



 
    