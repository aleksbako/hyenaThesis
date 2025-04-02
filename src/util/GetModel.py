import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from .Constants import EPOCH,VARMUP_EPOCH, HYENA_OPTIMIZER_TYPE , OPTIMIZER_TYPE
from .Train import train


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
      nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, nn.BatchNorm2d):
      nn.init.constant_(m.weight, 1)
      nn.init.constant_(m.bias, 0)



def get_model(baseline_model, model_type, train_dataloader,val_dataloader, loss, lr, weight_decay, epochs):
    best_accuracy = best_precision = 0
    try:
        print(f"{model_type}_checkpoint.pt")
        checkpoint = torch.load(f"../output/{model_type}_checkpoint.pt")
  
        
          # Load the state dictionary into the baseline_model
        baseline_model.load_state_dict(checkpoint['model'], strict=False)
        
        # Now model refers to the baseline_model with loaded weights
        model = baseline_model
        #optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        optim = getOptimizer(model, lr, weight_decay, model_type)
        #optim = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        epoch_times = checkpoint['epoch_times']
        epoch_val_times = checkpoint['epoch_val_times']
        mean_train_accuracy = checkpoint['mean_train_accuracy']
        mean_train_loss = checkpoint['mean_train_loss']
        mean_val_accuracy = checkpoint['mean_val_accuracy']
        mean_val_loss = checkpoint['mean_val_loss']
        best_accuracy = checkpoint['best_accuracy']
        best_precision = checkpoint['best_precision']
        

    except Exception as e:
        print(e)
        start_epoch = 0
        model = baseline_model
        #model.apply(init_weights)
        optim = getOptimizer(model, lr, weight_decay, model_type)
        #optim = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
        epoch_times = []
        mean_train_accuracy = []
        mean_train_loss = []
        mean_val_accuracy = []
        mean_val_loss = []
        epoch_val_times = []
    # Warm-up scheduler for the first 10 epochs
    #warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.1, total_iters=VARMUP_EPOCH) #0.001
    
    # Cosine annealing after warm-up
    #cosine_scheduler = CosineAnnealingLR(optim, T_max=EPOCH,eta_min=1e-5)  # Assuming total epochs = 100
    #scheduler =  torch.optim.lr_scheduler.SequentialLR(optim, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[VARMUP_EPOCH])

    scheduler = getScheduler(optim,model_type)
    print(f"The current epoch for {model_type} is: {start_epoch} ")
    if start_epoch < epochs:
        model = train(model,train_dataloader,val_dataloader,loss,optim,model_type, scheduler,epochs=epochs, start_epoch=start_epoch ,
                       epoch_times=epoch_times, epoch_val_times=epoch_val_times ,mean_train_loss=mean_train_loss, mean_train_accuracy=mean_train_accuracy ,
                         mean_val_loss=mean_val_loss, mean_val_accuracy=mean_val_accuracy, best_accuracy=best_accuracy, best_precision=best_precision)
    
    return model

def getOptimizer(model, lr, weight_decay, model_type):
    if "Hyena" in model_type:
        # Separate parameters for Hyena filters vs other parameters
        filter_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if 'filter_fn' in name or 'short_filter' in name:
                filter_params.append(param)
            else:
                other_params.append(param)
        
        # Use different learning rates for different components
        optimizer = torch.optim.AdamW([
            {'params': filter_params, 'lr': lr, 'weight_decay': weight_decay},
            {'params': other_params, 'lr': lr, 'weight_decay': weight_decay}
        ], betas=(0.9, 0.98))
        
        # Alternative SGD configuration if needed
        if HYENA_OPTIMIZER_TYPE == "SDG":
            optimizer = torch.optim.SGD([
                {'params': filter_params, 'lr': 1e-2, 
                 'weight_decay': 1e-2, 'momentum': 0.9},
                {'params': other_params, 'lr': lr, 
                 'weight_decay': weight_decay, 'momentum': 0.9}
            ])
    else:
        # Original optimizer for non-Hyena models
        if OPTIMIZER_TYPE == "SDG":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, 
                                      weight_decay=weight_decay, momentum=0.9)
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, 
                                        weight_decay=weight_decay)
    
    return optimizer


# Enhanced scheduler configuration
def getScheduler(optimizer, model_type):
    if "Hyena" in model_type:
        # More aggressive warmup for Hyena models
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=0.1,  # More conservative warmup
            total_iters=VARMUP_EPOCH
        )
        
        # Longer cosine annealing period
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=EPOCH,
            eta_min=1e-6  # Lower minimum learning rate
        )
        
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[VARMUP_EPOCH]
        )
    else:
        # Original scheduler for non-Hyena models
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=0.1, 
            total_iters=VARMUP_EPOCH
        )
        
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=EPOCH,
            eta_min=1e-5
        )
        
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[VARMUP_EPOCH]
        )
    
    return scheduler