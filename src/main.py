import torch
import torchvision
import torch.nn as nn
import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader , random_split
from datetime import datetime, timedelta

from HyenaOperator import HyenaOperator
from models.hyenaVit import HyenaVit
import numpy as np
import torch.nn.init as init
from models.ModifiedVit import ModifiedVit
from dataloaders.dataset.caltech256 import Caltech256Dataset
import time

def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            init.constant_(module.bias, 0)

def train(model, dataloader, loss, optimizer,model_type, warmup_epochs=5, checkpoint_interval=1, epochs=32, start_epoch=0, total_training_time=0):
    lr_init = optimizer.param_groups[0]['lr']
    lr_max = lr_init
    lr_min = lr_max / 10


    model.train()

    for epoch in range(start_epoch, epochs):
        current_losses = []
       
        start_time = time.time()

        # Calculate learning rate for the current epoch
        if epoch < warmup_epochs:
            lr = lr_init + (lr_max - lr_init) * epoch / warmup_epochs
        else:
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos((epoch - warmup_epochs) * np.pi / (16 - warmup_epochs)))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        for data in tqdm.tqdm(dataloader):
            optimizer.zero_grad()

            image = data[0].to("cuda")

            label = data[1].to("cuda")

            pred_label = model(image)

            current_loss = loss(pred_label, label)

            current_loss.backward()
            optimizer.step()
            current_losses.append(np.mean(current_loss.item()))

        end_time = time.time()
        epoch_training_time = end_time - start_time
        total_training_time += epoch_training_time

        print(f"{epoch}. epoch completed during training process with avg loss : {np.mean(current_losses)}, Total Training Time: {str(timedelta(seconds=total_training_time))}")

       
        if epoch % checkpoint_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'total_training_time': total_training_time
            }, f"../output/{model_type}_checkpoint.pt")

    return model
        
def get_model(baseline_model, model_type, train_dataloader,loss, lr, weight_decay, epochs):
   
    try:
        checkpoint = torch.load(f"../output/{model_type}_checkpoint.pt")
        start_epoch = checkpoint['epoch']
        model = baseline_model
        model.load_state_dict(checkpoint['model_state_dict'])
        optim = torch.optim.AdamW(ViT.parameters(), lr=lr,weight_decay=weight_decay)
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        total_training_time = checkpoint['total_training_time']
      

    
    except:
        start_epoch = 0
        model = baseline_model
        optim = torch.optim.AdamW(model.parameters(), lr=lr,weight_decay=weight_decay)
        total_training_time =0
        

    if start_epoch < epochs:
        
        model = train(model,train_dataloader,loss,optim,model_type, epochs=epochs, start_epoch=start_epoch , total_training_time=total_training_time)
    
    return model
    

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
    baseline = torchvision.models.vit_b_16(pretrained=False).to("cuda")
    ViT = ModifiedVit(baseline, 257).to("cuda")
    ViT.apply(init_weights)
    
    root_dir = "../data/256_ObjectCategories/"
    epochs = 32
    lr = (10**(-3))
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

    model = get_model(ViT, model_type, train_dataloader,loss, lr, weight_decay, epochs)

    hyena_ViT = HyenaVit().to("cuda")   
    hyena_ViT.apply(init_weights)
    hyenaLoss = nn.CrossEntropyLoss()
    hyenaLr=2*(10**(-4))
    heyena_weight_decay = 0.01

    model_type = "hyena"

    model_hyena = get_model(hyena_ViT, model_type, train_dataloader, hyenaLoss, hyenaLr, heyena_weight_decay, epochs)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    validate(model, val_dataloader, loss)
    validate(model_hyena, val_dataloader, hyenaLoss)
 

    
    

    



 
    