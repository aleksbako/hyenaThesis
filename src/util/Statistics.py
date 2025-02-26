import torch
import numpy as np

def get_median_time(model_type_1, model_type_2,output_dir="../output/"):
    try:

        model1_checkpoint = torch.load(f"../output/{model_type_1}_checkpoint.pt")
        
        model2_checkpoint = torch.load(f"../output/{model_type_2}_checkpoint.pt")

        print(f"Min time spent for {model_type_1} model : {np.min(model1_checkpoint['epoch_times'])}")
        print(f"Min time spent for {model_type_2} model : {np.min(model2_checkpoint['epoch_times'])}")

        # Adjust epoch times to be cumulative
        print(f"Median time spent for {model_type_1} model : {np.median(model1_checkpoint['epoch_times'])}")
        print(f"Median time spent for {model_type_2} model : {np.median(model2_checkpoint['epoch_times'])}")

        print(f"Max time spent for {model_type_1} model : {np.max(model1_checkpoint['epoch_times'])}")
        print(f"Max time spent for {model_type_2} model : {np.max(model2_checkpoint['epoch_times'])}")

        
        
        print(f"Median time spent in validation for {model_type_1} model : {np.median(model1_checkpoint['epoch_val_times'])}")
        print(f"Median time spent in validation for {model_type_2} model : {np.median(model2_checkpoint['epoch_val_times'])}")

        print(f"Median time spent in validation for {model_type_1} model : {np.median(model1_checkpoint['epoch_val_times'])}")
        print(f"Median time spent in validation for {model_type_2} model : {np.median(model2_checkpoint['epoch_val_times'])}")

        print(f"Max time spent in validation for {model_type_1} model : {np.max(model1_checkpoint['epoch_val_times'])}")
        print(f"Max time spent in validation for {model_type_2} model : {np.max(model2_checkpoint['epoch_val_times'])}")

    except:
        print("error when fetching  epoch time data")

