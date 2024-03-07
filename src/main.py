import torch
import torchvision
import torch.nn as nn
import tqdm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from datetime import datetime
from HyenaOperator import HyenaOperator
from hyenaVit import HyenaVit
import numpy as np
import torch.nn.init as init
from ModifiedVit import ModifiedVit
def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            init.constant_(module.bias, 0)

def train(model, dataloader, loss, optimizer, warmup_epochs=5):
        lr_init = optimizer.param_groups[0]['lr']
        lr_max = lr_init
        lr_min = lr_max / 10
        for epoch in range(16):
            current_losses = []
            model.train()

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
            print(f"{epoch}. epoch completed during training process with avg loss : {np.mean(current_losses)}")
            
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
    ViT = ModifiedVit(baseline, 100).to("cuda")
    ViT.apply(init_weights)


    lr = (10**(-3))
    loss = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(ViT.parameters(), lr=lr,weight_decay=0.05)
    
    batch_size = 64
    transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.Resize((224,224))
    ]
    )
    data = torchvision.datasets.CIFAR100("../data/", download=False, transform=transform,)
    train_dataloader = DataLoader(data,batch_size=batch_size, shuffle=True)
    
    try:
        model = torch.load("../output/model.pt")
    
    except:
        start_time = datetime.now()
        model = train(ViT,train_dataloader,loss,optim )
        end_time = datetime.now()
        print('Regular ViT Duration: {}'.format(end_time - start_time))
        torch.save(model,"../output/model.pt")

    hyenaViT = HyenaVit().to("cuda")
    hyenaViT.apply(init_weights)
    hyenaLoss = nn.CrossEntropyLoss()
    hyenaLr=2*(10**(-4))
    hyenaOptim = torch.optim.AdamW(hyenaViT.parameters(), lr=hyenaLr,weight_decay=0.01)

    try:
        model_hyena = torch.load("../output/hyena_model.pt")
    
    except:
  
        start_time = datetime.now()
        model_hyena = train(hyenaViT,train_dataloader,hyenaLoss,hyenaOptim )
        end_time = datetime.now()
        print('Hyena ViT Duration: {}'.format(end_time - start_time))
        torch.save(model_hyena,"../output/hyena_model.pt")

    named_modules_copy = dict(model_hyena.named_modules())
    for name, module in named_modules_copy.items():
        if isinstance(module, HyenaOperator):
            print(module)
    
    val_data = torchvision.datasets.CIFAR100("../data/", train=False, download=False, transform=transform)
    val_dataloader = DataLoader(val_data, batch_size=batch_size)
    validate(model, val_dataloader, loss)
    validate(model_hyena, val_dataloader, hyenaLoss)
 

    
    

    



 
    