import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from .Validation import validate
from datetime import  timedelta
from sklearn.metrics import classification_report
import re

def pretrain_unfreeze_all(model):
    # Freeze all layers except SE (attention layers)
    for name, param in model.named_parameters():
        param.requires_grad = True  # Freeze all other layers



def unfreeze_layers_SE(model, stage_num):
    """
    Gradually unfreezes layers in stages as training progresses.

    Args:
        model (torch.nn.Module): The model whose layers will be unfrozen.
        stage_num (int): Current stage (based on epoch number).
        total_stages (int): Total number of unfreezing stages.
    """
    # Freeze the entire model initially
    for param in model.parameters():
        param.requires_grad = False

    # Always unfreeze the classifier head if present
    if hasattr(model, 'head'):
        for param in model.head.parameters():
            print('Unfreezing head')
            param.requires_grad = True

    # Unfreeze stem layers at stage 1
    if stage_num > 0 and hasattr(model, 'stem'):
        for param in model.stem.parameters():
            print('Unfreezing stem layers')
            param.requires_grad = True

    # Get the model's stage layers
    if hasattr(model, 'stages'):
        stage_layers = list(model.stages)
        num_stages = len(stage_layers)

        if stage_num < 2:
            print("Skipping further unfreezing for early stages")
            return

        # Define the unfreezing rate based on the stage number
        if stage_num < 4:
            layers_to_unfreeze = stage_num - 1  # Unfreeze gradually at early stages
        else:
            layers_to_unfreeze = 4 + (stage_num - 4) * 2  # Faster unfreezing later

        # Ensure we don't exceed the number of stage layers
        layers_to_unfreeze = min(layers_to_unfreeze, num_stages)

        # Gradually unfreeze stage layers
        for i in range(layers_to_unfreeze):
            for param in stage_layers[i].parameters():
                print(f'Unfreezing stage {i + 1}')
                param.requires_grad = True

    # If stage_num is very high, unfreeze everything
    if stage_num >= num_stages+3:
        for param in model.parameters():
            print("Unfreezing all layers")
     
        

def unfreeze_layers(model, stage_num, total_stages):
    """
    Gradually unfreezes layers in stages as training progresses.

    Args:
        model (nn.Module): The transformer model.
        stage_num (int): Current stage (based on epoch number).
        total_stages (int): Total number of unfreezing stages.
    """
    # Freeze the entire model initially
    for param in model.parameters():
        param.requires_grad = False

    # Always unfreeze the classifier head

    for param in model.ViT.heads.parameters():
        print('unfreezing head')
        param.requires_grad = True

    # Unfreeze embeddings progressively (first step)
    if stage_num > 0:
        for param in model.ViT.conv_proj.parameters():
            print('unfreezing conv_proj')
            param.requires_grad = True
        for param in model.ViT.encoder.dropout.parameters():
            print('unfreezing dropout')
            param.requires_grad = True
        for param in model.ViT.encoder.ln.parameters():
            print('unfreezing ln')
            param.requires_grad = True
    # Get encoder layers
    encoder_layers = list(model.ViT.encoder.layers)
    num_encoder_layers = len(encoder_layers)

    if stage_num < 2:
        print("Skipping encoder unfreezing for early stages")
        return

        # Define the unfreezing rate based on the stage
    if stage_num < 8:  # Before epoch 40 (stages 0-7), unfreeze 1 layer per stage
        layers_to_unfreeze = stage_num - 1
    else:  # After epoch 40 (stage 8+), unfreeze 2 layers per stage
        layers_to_unfreeze = 5 + (stage_num - 5) * 2  # 7 layers before epoch 40, then 2 per stage
    


     # Ensure we don't exceed the number of encoder layers
    layers_to_unfreeze = min(layers_to_unfreeze, num_encoder_layers)
    

    # Gradually unfreeze encoder layers

    for i in range(layers_to_unfreeze):
        for param in encoder_layers[i].parameters():
            print(f'unfreezing encoder layer {i}')
            param.requires_grad = True


def unfreeze_layers_vit(model, stage_num):
    """
    Progressively unfreezes ViT layers for fine-tuning.

    Args:
        model (nn.Module): The Vision Transformer model.
        stage_num (int): Current training stage.
    """
    # Step 1: Freeze everything initially
    for param in model.parameters():
        param.requires_grad = False

    # Always unfreeze the classifier head first
    for param in model.ViT.heads.parameters():
        print('Unfreezing classifier head')
        param.requires_grad = True

    # Get the encoder layers
    encoder_layers = list(model.ViT.encoder.layers)
    num_encoder_layers = len(encoder_layers)

    if stage_num == 0:
        print("Stage 0: Training classifier head only")
        return

    elif stage_num == 1:
        print("Stage 1: Unfreezing mid-level encoder blocks (6-11)")
        for i in range(6, num_encoder_layers):
            for param in encoder_layers[i].parameters():
                param.requires_grad = True

    elif stage_num >= 2:
        print("Stage 2: Unfreezing all encoder layers")
        for i in range(num_encoder_layers):
            for param in encoder_layers[i].parameters():
                param.requires_grad = True

        # Optionally unfreeze the patch embedding layer
        for param in model.ViT.conv_proj.parameters():
            param.requires_grad = True

def unfreeze_self_attention_first(model, stage_num, total_stages):
    """
    First fine-tunes only the HyenaOperator layers of the encoders,
    then progressively unfreezes the rest of the model after the HyenaOperator is stabilized.

    Args:
        model (nn.Module): The transformer model.
        stage_num (int): Current stage (based on epoch number).
        total_stages (int): Total number of unfreezing stages.
    """
    # Freeze the entire model initially
    for param in model.parameters():
        param.requires_grad = False

    # Always unfreeze the classification head
    for param in model.ViT.heads.parameters():
        if not param.requires_grad:
            print('Unfreezing head')
            param.requires_grad = True

    # Get encoder layers
    encoder_layers = list(model.ViT.encoder.layers)
    num_encoder_layers = len(encoder_layers)

    # Stage 0: Unfreeze only the HyenaOperator in all encoder layers
    if stage_num == 0:
        for i in range(num_encoder_layers):
            for param in encoder_layers[i].self_attention.parameters():
                if not param.requires_grad:
                    print(f'Unfreezing HyenaOperator in encoder layer {i}')
                    param.requires_grad = True

    # Subsequent stages: Gradually unfreeze the rest of the model
    elif stage_num > 0:
        # Unfreeze 2 encoder layers per stage (including their HyenaOperator and other components)
        layers_to_unfreeze = min(stage_num, num_encoder_layers)
        for i in range(layers_to_unfreeze):
            for param in encoder_layers[i].parameters():  # Unfreeze entire encoder layer
                param.requires_grad = True

        # Unfreeze the patch embedding layer (conv_proj) after all encoder layers are unfrozen
        if stage_num >= num_encoder_layers/3:
            for param in model.ViT.conv_proj.parameters():
                if not param.requires_grad:
                    print('Unfreezing conv_proj')
                    param.requires_grad = True

def train(model, train_dataloader, val_dataloader, loss,
           optimizer,model_type,scheduler, epochs=32, start_epoch=0, epoch_times=[], epoch_val_times=[],
             mean_train_loss=[], mean_train_accuracy=[],mean_val_loss=[], mean_val_accuracy=[] ,best_accuracy=0, best_precision=0,unfreeze_stage_interval=10):

    train_losses = []
    val_losses = []
    prev_stage_num = -1
    #4 for SE and AA
    
    for epoch in range(start_epoch, epochs):
        stage_num = (epoch // unfreeze_stage_interval)  # Determines which stage we're at based on the epoch number
        
        if stage_num != prev_stage_num:
            
            if 'ViT' in model_type:
                #unfreeze_layers_vit(model,stage_num)
                #unfreeze_layers(model,stage_num, epochs // unfreeze_stage_interval)
                unfreeze_self_attention_first(model,stage_num, epochs // unfreeze_stage_interval)
                if(stage_num == 1):
                    unfreeze_stage_interval = 5
            elif 'Mobile' in model_type:
                pass
            else:
                unfreeze_layers_SE(model, stage_num)
                unfreeze_stage_interval=5  # Update layers based on the current stage
            prev_stage_num = stage_num  # Update the previous stage number    
            
        current_losses = []
        correct_predictions = 0
        total_samples = 0
        start_time = time.time()
        model.train()

        for batch_idx, data in enumerate(tqdm.tqdm(train_dataloader)):
          
            optimizer.zero_grad()
            
            image = data[0].to("cuda")
            
            label = data[1].to("cuda")
    
            pred_label = model(image)
            #pred_label = torch.clamp(pred_label, min=-1e6, max=1e6)  # Avoid extreme values

            current_loss = loss(pred_label, label)
            
            current_loss.backward()

            def log_param_changes(model, min_threshold=0.001, max_threshold=0.05):
                changes = {}
                
                # Store pre-update weights
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        changes[name] = {"before": param.clone().detach()}

                optimizer.step()  # Apply gradient update

                # Compare post-update weights
                for name, param in model.named_parameters():
                    if name in changes:
                        changes[name]["after"] = param.clone().detach()
                        change_magnitude = (changes[name]["after"] - changes[name]["before"]).norm().item()
                        
                        # Only print if change is too small or too large
                        if change_magnitude < min_threshold:
                            print(f"⚠️ Small Update: {name} | Change in Norm: {change_magnitude:.6f}")
                        elif change_magnitude > max_threshold:
                            print(f"🚨 Large Update: {name} | Change in Norm: {change_magnitude:.6f}")
                    
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            #if epoch % 2 == 1:
            optimizer.step()
            #else:
             #   log_param_changes(model)

            _, predicted = torch.max(pred_label, 1)
            correct_predictions += (predicted == label).sum().item()
            total_samples += label.size(0)
            
          #  if batch_idx % 100 == 99:  # Print every 100 batches
         #       print(f'Epoch [{epoch + 1}/{epochs}], Step [{batch_idx + 1}/{len(train_dataloader)}], '
    #  f'Loss: {current_loss.item():.4f}, Accuracy: {correct_predictions / total_samples:.4f}')
 
           
            current_losses.append(np.mean(current_loss.item()))
            #log_memory()
        end_time = time.time()
        epoch_training_time = end_time - start_time
        epoch_times.append(epoch_training_time)

        current_accuracy = correct_predictions / total_samples * 100
        mean_train_accuracy.append(current_accuracy)
        mean_train_loss.append(np.mean(current_losses))
        train_losses.append(np.mean(current_losses))

        print(f"{epoch}. Epoch completed during training process with current epoch loss : {np.mean(current_losses)}, accuracies: {current_accuracy} Total Training Time: {str(timedelta(seconds=np.sum(epoch_times)))}")

          # Validate the model after each epoch
        start_val_time = time.time()
        val_loss, val_accuracy, val_precision = validate(model, val_dataloader, loss)
        epoch_val_times.append(time.time()-start_val_time)
        val_losses.append(val_loss)

        mean_val_accuracy.append(val_accuracy)
        mean_val_loss.append(val_loss)
        
        # Update the best model based on validation accuracy and precision
        if val_accuracy > best_accuracy or val_precision > best_precision:
            best_accuracy = val_accuracy
            best_precision = val_precision
        
    
        scheduler.step()
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch_times': epoch_times,
            'epoch_val_times': epoch_val_times,
            'mean_train_accuracy': mean_train_accuracy,
            'mean_train_loss': mean_train_loss,
            'mean_val_accuracy': mean_val_accuracy,
            'mean_val_loss': mean_val_loss,
            'best_accuracy': best_accuracy,
            'best_precision': best_precision,
            }, f"../output/{model_type}_checkpoint.pt")
        
        filtered_train_losses = [loss if loss <= 7 else np.nan for loss in train_losses]
        filtered_val_losses = [loss if loss <= 7 else np.nan for loss in val_losses]
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(start_epoch, epoch+1), filtered_train_losses, label='Training Loss')
        plt.plot(range(start_epoch, epoch+1), filtered_val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'{model_type} Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"../output/{model_type}_loss_curve.png")
        plt.close()
      
    return model


def check_vanishing_gradients(model, threshold=1e-6):
    vanishing_layers = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_mean = param.grad.abs().mean().item()
            if grad_mean < threshold:
                vanishing_layers.append((name, grad_mean))
    
    if vanishing_layers:
        print("\n🔥 VANISHING GRADIENTS DETECTED:")
        for name, gmean in sorted(vanishing_layers, key=lambda x: x[1]):
            print(f"{name}: {gmean:.3e} (Below threshold {threshold:.1e})")
        return True
    return False
def revive_gradients(model, scale=1e-3):
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().max() < 1e-6:
            # Inject directional noise
            noise = torch.randn_like(param.grad) * scale
            param.grad += noise
            param.grad = torch.where(
                param.grad.abs() < 1e-6,
                param.grad * 1000,  # Forcefully amplify tiny grads
                param.grad
            )
           # print(f"Revived gradients for {name}")

#def log_memory():
#    print(f"CPU: {psutil.virtual_memory().percent}% | "
#          f"GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")
