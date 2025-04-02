import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader , random_split
import numpy as np
import os
from sklearn.model_selection import ParameterGrid

from dataloaders.dataset.caltech256 import Caltech256Dataset
from dataloaders.dataset.ImageNetValDataset import ImageNetValDataset
from models.hyenaVit import HyenaVit
from models.Vit import Vit
from models.SimpleHyenaViT import SimpleHyenaViT

from util.Lime import calculate_Lime
from util.Visualization import plot_metrics
from util.GradCAM import calculate_Grad_CAM
from util.GradCAMViT import calculate_vit_grad_cam
from dataloaders.ImageNetLoader import getImageNetDataLoaders
from dataloaders.Caltech256Loader import getCaltechDataLoaders
from dataloaders.Cifar100Loader import getCifarDataLoaders
from util import validate, train, get_model, get_median_time, OUTPUT_DIR, DATA_DIR, DEVICE
from util import IMAGE_SIZE, BATCH_SIZE, LOSS, EPOCH, MEAN, STD, NUM_CLASSES, VAL_TRANSFORMATION,TRAIN_TRANSFORMATION
from util import LEARNING_RATE, WEIGHT_DECAY
from util import HYENA_LEARNING_RATE, HYENA_WEIGHT_DECAY, HYENA_LOSS
from experiments.layer_change import ViT_experiments, SE_experiments, test_SE, AA_experiments, MobileNet_experiments


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

def cross_validate(train_dataset, val_dataset, param_grid, loss_function, epochs, batch_size=64):
    best_model = None
    best_score = 0
    best_params = None
    results = []
    output_dir = "../output/"
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, "cross_validation_results.txt")
    # Create DataLoaders
    mean=MEAN
    std=STD
 
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
    #1, 25, 90
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    img_paths = [ (DATA_DIR+'256_ObjectCategories/164.porcupine/164_0006.jpg',163,[128,151,212]),
        (DATA_DIR+'256_ObjectCategories/145.motorbikes-101/145_0430.jpg',144,[104,145,250]) ]
        
                # (DATA_DIR+'256_ObjectCategories/015.bonsai-101/015_0015.jpg',14),
                # (DATA_DIR+'256_ObjectCategories/230.trilobite-101/230_0074.jpg',229), 
               #  (DATA_DIR+'256_ObjectCategories/252.car-side-101/252_0034.jpg',251)]




    #cal_dataset = Caltech256Dataset(DATA_DIR + "256_ObjectCategories/",transform=train_transform)
    #cal_dataset = torchvision.datasets.Caltech256(root="../data/",download=False)
        # Calculate the sizes for the splits
    #train_size = int(0.85 * len(cal_dataset))
    #val_size = int(0.1 * len(cal_dataset))
   # test_size = len(cal_dataset) - train_size - val_size  # Remaining data for testing

    #train_dataset, val_dataset, test_dataset = random_split(cal_dataset, [train_size, val_size,test_size], torch.Generator().manual_seed(11))

    # Create data loaders
    #train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
   # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    #test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Apply the respective transformations
   # train_dataset.dataset.transform = train_transform
    #val_dataset.dataset.transform = val_transform
    #train_loader, val_loader, dataset  = getImageNetDataLoaders(train_transform, VAL_TRANSFORMATION)
    train_loader, val_loader, dataset = getCaltechDataLoaders(TRAIN_TRANSFORMATION, VAL_TRANSFORMATION)
    #train_loader, val_loader, dataset = getCifarDataLoaders(TRAIN_TRANSFORMATION, VAL_TRANSFORMATION)
    #ViT_experiments(dataset,train_loader, val_loader,img_paths)
    #SE_experiments(dataset,train_loader, val_loader,img_paths)
    #MobileNet_experiments(dataset,train_loader, val_loader,img_paths)
    AA_experiments(dataset,train_loader, val_loader,img_paths)
    #test_SE(cal_dataset,train_loader, val_loader)
    #ViT = Vit(preTrained=True).to(DEVICE)
   # model = get_model(ViT, "ViT", train_loader, val_loader, LOSS, LEARNING_RATE, WEIGHT_DECAY, EPOCH)
   #


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
    

   




    
    

    



 
    