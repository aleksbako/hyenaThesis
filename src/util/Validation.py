import torch
import tqdm
from sklearn.metrics import accuracy_score, precision_score

def validate(model, dataloader, loss):
    model.eval()
    total_samples = 0
    total_loss = 0
    correct_predictions = 0
    all_predicted = []  # Accumulate predictions for the entire epoch
    all_labels = []     # Accumulate ground truth labels for the entire epoch

    with torch.no_grad():
        for data in tqdm.tqdm(dataloader):
            image, labels = data
            
            image = image.to("cuda")
            
            if isinstance(labels,torch.Tensor) == False:
                labels = torch.tensor(list(labels), dtype=torch.long)
           
            labels = labels.to("cuda")

            pred_label = model(image)
            current_loss = loss(pred_label, labels)
            total_loss += current_loss.item()
          
            _, predicted = torch.max(pred_label, 1)
            correct_predictions += (predicted == labels).sum().item()

            total_samples += labels.size(0)
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
 
        
    average_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples * 100
    precision = precision_score(all_labels, all_predicted, average='weighted', zero_division=0.0) * 100
    
    #report = classification_report(all_labels, all_predicted)
   # print(report)

    print(f"Validation Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}%")
    return average_loss , accuracy, precision