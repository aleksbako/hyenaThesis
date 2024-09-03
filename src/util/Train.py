import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from Validation import validate

def train(model, train_dataloader, val_dataloader, loss, optimizer,model_type,scheduler, epochs=32, start_epoch=0, epoch_times=[], mean_loss=[], mean_accuracy=[],best_accuracy=0, best_precision=0):
    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    train_losses = []
    val_losses = []
    
    for epoch in range(start_epoch, epochs):
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

            current_loss = loss(pred_label, label)
            current_loss.backward()
            optimizer.step()

            _, predicted = torch.max(pred_label, 1)
            correct_predictions += (predicted == label).sum().item()
            total_samples += label.size(0)

            if batch_idx % 100 == 99:  # Print every 100 batches
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{batch_idx + 1}/{len(train_dataloader)}], '
      f'Loss: {current_loss.item():.4f}, Accuracy: {correct_predictions / total_samples:.4f}')
                

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
        