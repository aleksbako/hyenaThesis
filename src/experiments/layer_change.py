import torch
from util.GetModel import get_model
from util import DEVICE, OUTPUT_DIR,EPOCH, NUM_CLASSES, IMAGE_SIZE
from util.GradCAMViT import calculate_vit_grad_cam
from util.GradCAM import calculate_Grad_CAM
from util.Lime import calculate_Lime
from util import LOSS, LEARNING_RATE, WEIGHT_DECAY , HYENA_LOSS, HYENA_LEARNING_RATE, HYENA_WEIGHT_DECAY
from util import validate
from util.Statistics import get_median_time
from util.Visualization import plot_metrics
from models.Vit import Vit
from models.MobileNetWithHyena import MobileNetWithHyena
from models.MobileNetWithSE import MobileNetWithSE
from models.hyenaVit import HyenaVit
from models.SERestNet50 import SEResNet50
from Hyena.SEBasicHyenaBlock import SEBasicHyenaBlock
from modules.SEBottleneck import SEBottleneck
from Hyena.Hyena_SEBottleneck import Hyena_SEBottleneck
from Hyena.Hyena_SEBlock import SE_Block_Hyena
from modules.SEBasicBlock import SEBasicBlock
from Hyena.HyenaOperator import HyenaOperator
from Hyena.HyenaTransformerBlock import HyenaTransformerBlock
from Hyena.Hyena_WideBasic import Hyena_WideBasic
from util.LearningRateFinder import LearningRateFinder
from util.Shap import caluclateShap
from util.Counterfactual import counterfactual, class_transformation_experiment , plot_frequency_analysis ,compare_feature_importance, decision_boundary_analysis
import torch.nn as nn
import torchvision
import timm
from modules.WideBasic import WideBasic
import re
import torch.nn.init as init
import os

def pretrain_freeze_SE(SE, type=None):
    to_replace = []
    # Freeze all layers except SE (attention layers)
    for name, param in SE.named_parameters():
        if not name.endswith('attn') and 'head' not in name:  # 'se' is often used in the name of Squeeze-and-Excitation layers
            param.requires_grad = False  # Freeze all other layers
    if(type == 'hyena'):
        for name, module in SE.named_modules():
           
            if isinstance(module, nn.Module) and name.endswith('attn'):
                print(f'Found {name} for replacement')
                to_replace.append(name)
        for name in to_replace:
            #print(name)
            module = SE.get_submodule(name)
          

            print(f'Replacing {name} with SEBasicHyenaBlock')

                    # Retrieve the input and output channels of the attention layer
                
                        # Check for attributes specific to the attention layers
            in_channels = module.fc1.in_channels
            out_channels = module.fc2.out_channels 


                        # Create a new Hyena block
            new_hyena_block = SE_Block_Hyena(in_channels, out_channels,filter_order=64)
            init_hyena_operator_params(new_hyena_block.excitation)

                        # Replace the attention layer in the model
             # Replace the attention layer in the model
            parent_module = SE
            for attr in name.split('.')[:-1]:  # Navigate to the parent module
                parent_module = getattr(parent_module, attr)
            #  print('--------------------------------------------------------------------------------------------')
            #print("TEST")
        
            #print(name.split('.')[-1])
            #print(parent_module)
            setattr(parent_module, name.split('.')[-1], new_hyena_block)  # Repla
            #print(parent_module)
            #print('--------------------------------------------------------------------------------------------')
        # Check the replacement
        



    # Reset the weights of the attention (SE) layers
    for name, module in SE.named_modules():
        if name.endswith('attn') :  # Only target SE layers
               # print(f'Replaced {name} with SEBasicHyenaBlock')
            for param in module.parameters():
                #if param.dim() > 1:  # For Conv layers
                    #torch.nn.init.kaiming_normal_(param)
                if param.dim() == 1:  # For BatchNorm biases
                    torch.nn.init.normal_(param, mean=1, std=0.02)
                else:  # For other layers, zero init (biases)
                    torch.nn.init.zeros_(param)
    for name, param in SE.named_parameters():
        if 'attn' in name or 'head' in name: 
            param.requires_grad = True

def init_hyena_operator_params(module):
    """Initialize all parameters in the HyenaOperator with Hyena-specific schemes."""
    # GELU-specific gain factor
    GELU_GAIN = 1.414  # sqrt(2) approximation that works well with GELU
    
    # Initialize linear projections
    if hasattr(module, 'in_proj'):
        nn.init.xavier_normal_(module.in_proj.weight, gain=GELU_GAIN)
        if hasattr(module.in_proj, 'bias') and module.in_proj.bias is not None:
            nn.init.normal_(module.in_proj.bias, mean=0.0, std=0.01)

    if hasattr(module, 'out_proj'):
        nn.init.xavier_normal_(module.out_proj.weight, gain=1.0)  # Conservative output
        if hasattr(module.out_proj, 'bias') and module.out_proj.bias is not None:
            nn.init.zeros_(module.out_proj.bias)
    
    # Initialize short filter convolution
    if hasattr(module, 'short_filter'):
        if hasattr(nn.init, 'dirac_'):
            nn.init.dirac_(module.short_filter.weight)
        else:
            # Fallback for older PyTorch versions
            with torch.no_grad():
                module.short_filter.weight.fill_(0)
                for i in range(min(module.short_filter.weight.shape[:2])):
                    module.short_filter.weight[i,i] = 1.0
        
        if hasattr(module.short_filter, 'bias') and module.short_filter.bias is not None:
            nn.init.constant_(module.short_filter.bias, 0.01)

    # Specialized initialization for HyenaFilter components
    if hasattr(module, 'filter_fn'):
        # Initialize positional embeddings
        if hasattr(module.filter_fn, 'pos_emb') and hasattr(module.filter_fn.pos_emb, 'z'):
            nn.init.normal_(module.filter_fn.pos_emb.z, std=0.1)
        
        # Initialize implicit filter MLP
        if hasattr(module.filter_fn, 'implicit_filter'):
            for layer in module.filter_fn.implicit_filter:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=GELU_GAIN)
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        nn.init.normal_(layer.bias, mean=0.1, std=0.01)
        
        # Initialize modulation
        if hasattr(module.filter_fn, 'modulation'):
            for name, param in module.filter_fn.modulation.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param, gain=0.5)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.1)
        
        # Initialize bias
        if hasattr(module.filter_fn, 'bias'):
            nn.init.normal_(module.filter_fn.bias, mean=0.1, std=0.01)

def pretrain_freeze_AA(SE, type=None):
    to_replace = []
    # Freeze all layers except SE (attention layers)
   # for name, param in SE.named_parameters():
        #if re.match(r'^stages\.\d+\.\d+$', name) == False and 'head' not in name :  # 'se' is often used in the name of Squeeze-and-Excitation layers
       #     param.requires_grad = False  # Freeze all other layers

    for name, module in SE.named_modules():
           
        if re.match(r'^stages\.\d+\.\d+$', name):
     
            print(f'Found {name} for replacement')
            to_replace.append(name)

    for name in to_replace:
            #print(name)
        module = SE.get_submodule(name)
          

        print(f'Replacing {name} with SEBasicHyenaBlock')

                    # Retrieve the input and output channels of the attention layer
                
                        # Check for attributes specific to the attention layers
        #print(module)
        
        in_channels, out_channels = get_input_output_from_bottleneck(module)

        if type=='hyena':
                        # Create a new Hyena block
            new_block = Hyena_WideBasic(in_channels, out_channels, 0.3,16)
            init_hyena_operator_params(new_block.conv1.hyena)
            init_hyena_operator_params(new_block.conv2.hyena)
        else:
            new_block = WideBasic(in_channels,out_channels,0.4,shape=16).to('cuda')
                        # Replace the attention layer in the model
             # Replace the attention layer in the model
        parent_module = SE
        for attr in name.split('.')[:-1]:  # Navigate to the parent module
            parent_module = getattr(parent_module, attr)
            #  print('--------------------------------------------------------------------------------------------')
            #print("TEST")
        
            #print(name.split('.')[-1])
            #print(parent_module)

        setattr(parent_module, name.split('.')[-1], new_block)  # Repla
            #print(parent_module)
            #print('--------------------------------------------------------------------------------------------')
        # Check the replacement

        # Reset the weights of the attention (SE) layers
    for name, module in SE.named_modules():

        if re.match(r'^stages\.\d+\.\d+$', name) : 
            # Only target SE layers

               # print(f'Replaced {name} with SEBasicHyenaBlock')
            for param in module.parameters():
               
                if param.requires_grad: 
                    # Ensure it is trainable
    
                   # if param.dim() > 1:  # For Conv layers
                    #    torch.nn.init.kaiming_normal_(param)
                    if param.dim() == 1:  # For BatchNorm biases
                        torch.nn.init.normal_(param, mean=1, std=0.02)
                    elif param.dim() < 1:  # For other layers, zero init (biases)
                        torch.nn.init.zeros_(param)
    for name, param in SE.named_parameters():
        if re.match(r'^stages\.\d+\.\d+$', name) or 'head' in name : 
            param.requires_grad = True



def replace_attention_with_hyena(model, model_type=None):
    to_replace = []
    # Identify all MultiheadAttention layers
    count = 0
    for name, module in model.named_modules():
        if re.fullmatch(r'.*\.encoder_layer_\d+.self_attention', name):
            if count > 10:
                break
            print(f'Found Transformer block: {name}')
            count += 1
            to_replace.append(name)

    for name, param in model.named_parameters():
        if 'self_attention' not in name and 'head' not in name:  
            pass
            #param.requires_grad = False

    if model_type == 'hyena':
        #print(to_replace)
            # Replace EncoderBlock layers with HyenaTransformerBlock
        for name in to_replace:
            # Retrieve the module to be replaced
            
            module = model.get_submodule(name)
            #print(module)
            embed_dim = module.out_proj.in_features

                #out_channels = embed_dim  # Assuming output dims match the embed_dim for attention layers
            """
            # Create the HyenaTransformerBlock
            new_hyena_module = HyenaTransformerBlock(
                    d_model=embed_dim, 
                    l_max=197,
                    mlp_dim=module.mlp[0].out_features,  # ViT-B16 has 197 tokens (196 patches + CLS token)
                    dropout=0.5  # Same dropout as ViT
            ).to(next(model.parameters()).device)
            """
                    # Create the HyenaTransformerBlock
            new_hyena_module = HyenaOperator(
                    d_model=embed_dim, 
                    l_max=197,
                    filter_order=32,
                    filter_dropout=0.3,
                    dropout=0.3,
                    isVit=True  # Same dropout as ViT
            ).to(next(model.parameters()).device)
            init_hyena_operator_params(new_hyena_module)
            

            # Now directly replace the layer
            parent_module = model
                #parent_module = getattr(parent_module, "Vit."+name)
            parts = name.split('.')
               # print(name)
            target_layer_name = parts[-1]
            for i, part in enumerate(parts[:-1]):  # Traverse all but the last part (which is the layer to replace)
                parent_module = getattr(parent_module, part)  # Get the next submodule
                    #print(f"Traversing to: {part}")  # Debugging: Show where we are in the model

                    # Check the next part and print it for matching
                if i + 1 < len(parts):
                    next_part = parts[i + 1]
                        #print(f"Next part to check: {next_part}") 
                       
                    if(next_part == target_layer_name):
                           #print("MATCHING")
                        break
                                
            #print(parent_module)
            #print("Target layer : "  + target_layer_name)
            setattr(parent_module, target_layer_name, new_hyena_module)
               # print(f"Replaced {name.split('.')[-1]} with new Hyena layer")
        # Reset the weights of the attention (SE) layers
    for name, module in model.named_modules():

        if  re.fullmatch(r'.*\.encoder_layer_\d+.self_attention', name):# or 'head' in name : #re.fullmatch(r'.*\.encoder_layer_\d+.self_attention', name) or  # Only target SE layers
           # return model

               # print(f'Replaced {name} with SEBasicHyenaBlock')
            for param_name, param in module.named_parameters():
               
                if param.requires_grad:  # Ensure it is trainable
           
                    if param.dim() > 1:  # Weights (e.g., Linear, Conv layers)
                        nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                    elif param.dim() == 1:  # Biases (e.g., Linear, Conv, BatchNorm)
                        if 'bias' in param_name:  # Initialize biases to zero
                            nn.init.zeros_(param)
                        elif 'weight' in param_name and isinstance(module, nn.BatchNorm1d):  # BatchNorm weights
                            nn.init.ones_(param)
                        elif 'weight' in param_name and isinstance(module, nn.LayerNorm):  # LayerNorm weights
                            nn.init.ones_(param)
                        else:
                        # Handle other 1D parameters (e.g., custom layers)
                            nn.init.normal_(param, mean=0, std=0.02)  # Example initialization

    # Freeze all parameters except the HyenaOperator layers
 

    return model



def get_input_output_from_bottleneck(block):
    # Check if the block is inside a Sequential
   # print(block)
    if isinstance(block, nn.Module):
        if isinstance(block.shortcut, torch.nn.Identity):
        # If shortcut is Identity, it just passes the input without modification
            first_input_channels = block.conv1_1x1.conv.in_channels  # Get input channels from the first conv layer
        else:
        # If shortcut is not Identity, get input channels from the shortcut layer (e.g., via conv)
            first_input_channels = block.shortcut.conv.in_channels
    
        
        # The last layer (conv3_1x1) is at index 3 in the Sequential container
    
        last_output_channels = block.conv3_1x1.out_channels
    else:
        first_input_channels = None
        last_output_channels = None
    
    return first_input_channels, last_output_channels

def initialize_weights(model):
    """
    Initialize the weights of the model using Kaiming initialization.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            # Kaiming initialization for convolutional layers
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            # Initialize BatchNorm weights to 1 and biases to 0
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # Kaiming initialization for fully connected layers
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
           
def ViT_experiments(dataset, train_loader, val_loader, listOfImages):
    ViT = Vit(preTrained=True,classNumber=NUM_CLASSES).to(DEVICE)
    ViT = replace_attention_with_hyena(ViT, 'model')

    model = get_model(ViT, "ViT", train_loader, val_loader, LOSS, LEARNING_RATE, WEIGHT_DECAY, EPOCH)
    
    hyena_ViT =  Vit(preTrained=True,classNumber=NUM_CLASSES).to(DEVICE) 
    
    hyena_ViT = replace_attention_with_hyena(hyena_ViT, 'hyena').to('cuda')
    #initialize_weights(hyena_ViT)

    model_hyena = get_model(hyena_ViT, "ViT_with_Hyena", train_loader,val_loader, HYENA_LOSS, HYENA_LEARNING_RATE, HYENA_WEIGHT_DECAY, EPOCH)
    
    #input_tensor = torch.randn(1, 3, 224, 224).to(DEVICE)   # Example input

    # Measure attention-based ViT
    #print("Attention ViT VRAM usage:")
    #attention_mem = measure_vram_usage(model, input_tensor)
    #print(f"{attention_mem:.2f} MB")

    # Measure Hyena-based ViT
    #print("\nHyena ViT VRAM usage:")
    #hyena_mem = measure_vram_usage(model_hyena, input_tensor)
    #print(f"{hyena_mem:.2f} MB")

    #validate(model, train_loader, LOSS)
    #validate(model, val_loader, LOSS)
    #validate(model_hyena, train_loader, HYENA_LOSS)
    #validate(model_hyena, val_loader, HYENA_LOSS)
    
    #plot_metrics("ViT", "ViT_with_Hyena",output_dir=OUTPUT_DIR)
    #get_median_time("ViT", "ViT_with_Hyena")
    #print(model)
    target_layer = [model.ViT.encoder.layers[0].ln_2,model.ViT.encoder.layers[len(model_hyena.ViT.encoder.layers)//2].ln_2,model.ViT.encoder.layers[-2].ln_2]
   # print(model_hyena)
    hyena_target_layer = [model_hyena.ViT.encoder.layers[0].ln_2,model_hyena.ViT.encoder.layers[len(model_hyena.ViT.encoder.layers)//2].ln_2 ,model_hyena.ViT.encoder.layers[-2].ln_2]
    
    for imagepath, label, target_classes in listOfImages:
        
       # counterfactual(model,imagepath, label)
        #counterfactual(model_hyena,imagepath, label)
   
        
        #calculate_vit_grad_cam(model, target_layer, imagepath, "ViT" ,"vit_base_patch16")
        #calculate_vit_grad_cam(model, target_layer, imagepath, "ViT" ,"vit_base_patch16", isPerturbed=True, perturbation_type='noise', intensity=0.1)
        #calculate_vit_grad_cam(model, target_layer, imagepath, "ViT" ,"vit_base_patch16", isPerturbed=True, perturbation_type='blur', intensity=0.6)
        #calculate_vit_grad_cam(model, target_layer, imagepath, "ViT" ,"vit_base_patch16", isPerturbed=True, perturbation_type='shift', intensity=0.1)
        calculate_vit_grad_cam(model, target_layer, imagepath, "ViT" ,"vit_base_patch16", intensity=0.001,FGSM=True)

        calculate_vit_grad_cam(model_hyena, hyena_target_layer, imagepath, "ViT_with_hyena" ,"hyena_vit_base_patch16",intensity=0.001,FGSM=True)
        #calculate_vit_grad_cam(model_hyena, hyena_target_layer, imagepath, "ViT_with_hyena" ,"hyena_vit_base_patch16")
        #calculate_vit_grad_cam(model_hyena, hyena_target_layer, imagepath, "ViT_with_hyena" ,"hyena_vit_base_patch16", isPerturbed=True, perturbation_type='noise', intensity=0.1)
        #calculate_vit_grad_cam(model_hyena, hyena_target_layer, imagepath, "ViT_with_hyena" ,"hyena_vit_base_patch16", isPerturbed=True, perturbation_type='blur', intensity=0.6)
        #calculate_vit_grad_cam(model_hyena, hyena_target_layer, imagepath, "ViT_with_hyena" ,"hyena_vit_base_patch16", isPerturbed=True, perturbation_type='shift', intensity=0.1)

        #compare_feature_importance(
        #    attention_model=model,
        #    hyena_model=model_hyena,
        #    image_path=imagepath,
        #    original_class=label,
        #    target_classes=target_classes
        #)

        #decision_boundary_analysis( attention_model=model,
        #   hyena_model=model_hyena,
        #    image_path=imagepath,
        #    original_class=label,
        #    target_classes=target_classes
        #)
        #hyena_result = counterfactual(model_hyena, imagepath, target_class=target_classes[0]) #class_transformation_experiment(model_hyena,imagepath,label,target_classes)
        #attn_result = counterfactual(model, imagepath, target_class=target_classes[0]) #class_transformation_experiment(model,imagepath,label,target_classes)
        #attn_diff = attn_result['visualizations']["normalized_difference"]  # Shape: (H,W,3)
        #hyena_diff = hyena_result['visualizations']["normalized_difference"]
        #plot_frequency_analysis(attn_diff, hyena_diff)




def SE_experiments(dataset, train_loader, val_loader, listOfImages):
    #SE = timm.create_model('seresnet33ts.ra2_in1k', pretrained=True, num_classes=NUM_CLASSES)
    #pretrain_freeze_SE(SE)
   
    #SE = SEResNet50(SEBasicBlock,NUM_CLASSES,False, dropout_rate=0.6).to(DEVICE)
    #SE = torchvision.models.resnet18()
    #SE.fc = nn.Linear(SE.fc.in_features, 257)
    #SE.to(DEVICE)
    
   # model = get_model(SE, "SE", train_loader, val_loader, LOSS, LEARNING_RATE, WEIGHT_DECAY, EPOCH)

    SE_Hyena = timm.create_model('seresnet33ts.ra2_in1k', pretrained=True, num_classes=NUM_CLASSES)

    pretrain_freeze_SE(SE_Hyena, "hyena")
    print(SE_Hyena)
    SE_Hyena.to(DEVICE)

    model_hyena = get_model(SE_Hyena, "SE_with_Hyena", train_loader,val_loader, HYENA_LOSS, HYENA_LEARNING_RATE, HYENA_WEIGHT_DECAY, EPOCH)

    #validate(model, val_loader, LOSS)
    validate(model_hyena, val_loader, HYENA_LOSS)
    
    #plot_metrics("SE", "SE_with_Hyena",output_dir=OUTPUT_DIR)
    #get_median_time("SE", "SE_with_Hyena")
    
    #check_class_predictions_by_index(model, train_loader, NUM_CLASSES, DEVICE ,"SE", "training")
    #check_class_predictions_by_index(model, val_loader, NUM_CLASSES, DEVICE, "SE", "validation")
    #check_class_predictions_by_index(model_hyena, train_loader, NUM_CLASSES, DEVICE, "SE_Hyena", "training")
    #check_class_predictions_by_index(model_hyena, val_loader, NUM_CLASSES, DEVICE, "SE_Hyena", "validation")
    #target_layer = [model.stages[0][1].attn, model.stages[1][1].attn,model.stages[2][1].attn,model.stages[3][1].attn]
    hyena_target_layer = [model_hyena.stages[0][1].attn,model_hyena.stages[1][1].attn,model_hyena.stages[2][1].attn,model_hyena.stages[3][1].attn]

    #print(model_hyena)
    for imagepath, label, target_classes in listOfImages:
        
        #calculate_Grad_CAM(model, target_layer, imagepath, "SE","SE")
       # calculate_Grad_CAM(model, target_layer, imagepath, "SE","SE", isPerturbed=True, perturbation_type='noise', intensity=0.05)
        #calculate_Grad_CAM(model, target_layer, imagepath, "SE","SE", isPerturbed=True, perturbation_type='blur', intensity=0.5)
        #calculate_Grad_CAM(model, target_layer, imagepath, "SE","SE", isPerturbed=True, perturbation_type='shift', intensity=0.1)


        #calculate_Grad_CAM(model_hyena, hyena_target_layer, imagepath, "SE_with_Hyena" ,"SE_Hyena")
        #calculate_Grad_CAM(model_hyena, hyena_target_layer, imagepath, "SE_with_Hyena" ,"SE_Hyena",isPerturbed=True,intensity=0.05)
        #calculate_Grad_CAM(model_hyena, hyena_target_layer, imagepath, "SE_with_Hyena" ,"SE_Hyena",isPerturbed=True, perturbation_type='blur',intensity=0.5)
        #calculate_Grad_CAM(model_hyena, hyena_target_layer, imagepath, "SE_with_Hyena" ,"SE_Hyena",isPerturbed=True,perturbation_type='shift',intensity=0.1)
  
 
        #counterfactual(model,imagepath, label)
        #counterfactual(model_hyena,imagepath, label)

        #compare_feature_importance(
        #    attention_model=model,
        #    hyena_model=model_hyena,
        #    image_path=imagepath,
        #    original_class=label,
        #    target_classes=target_classes
        #)

        #decision_boundary_analysis( attention_model=model,
        #   hyena_model=model_hyena,
        #    image_path=imagepath,
        #    original_class=label,
        #    target_classes=target_classes
        #)
        hyena_result = counterfactual(model_hyena, imagepath, target_class=target_classes[0]) #class_transformation_experiment(model_hyena,imagepath,label,target_classes)
        attn_result = counterfactual(model, imagepath, target_class=target_classes[0]) #class_transformation_experiment(model,imagepath,label,target_classes)
        attn_diff = attn_result['visualizations']["normalized_difference"]  # Shape: (H,W,3)
        hyena_diff = hyena_result['visualizations']["normalized_difference"]
        plot_frequency_analysis(attn_diff, hyena_diff)
        
    #caluclateShap(model,val_loader, dataset.categories, target_layer, listOfImages)
    #caluclateShap(model_hyena,val_loader, dataset.categories, hyena_target_layer, listOfImages)


def MobileNet_experiments(dataset, train_loader, val_loader, listOfImages):
    MobileSE = MobileNetWithSE(NUM_CLASSES, 3, IMAGE_SIZE)
    initialize_weights(MobileSE)
    MobileSE.to(DEVICE)
    
    model = get_model(MobileSE, "MobileSE", train_loader, val_loader, LOSS, LEARNING_RATE, WEIGHT_DECAY, EPOCH)

    MobileSE_Hyena = MobileNetWithHyena(NUM_CLASSES,3, IMAGE_SIZE)
    initialize_weights(MobileSE_Hyena)
    MobileSE_Hyena.to(DEVICE)
    model_hyena = get_model(MobileSE_Hyena, "MobileSE_Hyena", train_loader,val_loader, HYENA_LOSS, HYENA_LEARNING_RATE, HYENA_WEIGHT_DECAY, EPOCH)

    validate(model, val_loader, LOSS)
    validate(model_hyena, val_loader, HYENA_LOSS)
    
    #plot_metrics("MobileSE", "MobileSE_Hyena",output_dir=OUTPUT_DIR)
    #get_median_time("MobileSE", "MobileSE_Hyena")
    
    #check_class_predictions_by_index(model, train_loader, NUM_CLASSES, DEVICE ,"SE", "training")
    #check_class_predictions_by_index(model, val_loader, NUM_CLASSES, DEVICE, "SE", "validation")
    #check_class_predictions_by_index(model_hyena, train_loader, NUM_CLASSES, DEVICE, "SE_Hyena", "training")
    #check_class_predictions_by_index(model_hyena, val_loader, NUM_CLASSES, DEVICE, "SE_Hyena", "validation")
    #target_layer = [model.stages[0][1].attn, model.stages[1][1].attn,model.stages[2][1].attn,model.stages[3][1].attn]
    #hyena_target_layer = [model_hyena.stages[0][1].attn,model_hyena.stages[1][1].attn,model_hyena.stages[2][1].attn,model_hyena.stages[3][1].attn]

    #print(model_hyena)
    for imagepath, label in listOfImages:
        pass
        #calculate_Grad_CAM(model, target_layer, imagepath, "SE","SE")
        #calculate_Grad_CAM(model, target_layer, imagepath, "SE","SE", isPerturbed=True, perturbation_type='noise', intensity=0.04)
        #calculate_Grad_CAM(model, target_layer, imagepath, "SE","SE", isPerturbed=True, perturbation_type='blur', intensity=1.1)
        #calculate_Grad_CAM(model, target_layer, imagepath, "SE","SE", isPerturbed=True, perturbation_type='shift', intensity=0.1)
     
       # calculate_Lime(model, imagepath, DEVICE, dataset, "SE")

        #calculate_Grad_CAM(model_hyena, hyena_target_layer, imagepath, "SE_with_Hyena" ,"SE_Hyena")
        #calculate_Grad_CAM(model_hyena, hyena_target_layer, imagepath, "SE_with_Hyena" ,"SE_Hyena",isPerturbed=True,intensity=0.04)
        #calculate_Grad_CAM(model_hyena, hyena_target_layer, imagepath, "SE_with_Hyena" ,"SE_Hyena",isPerturbed=True, perturbation_type='blur',intensity=1.1)
        #calculate_Grad_CAM(model_hyena, hyena_target_layer, imagepath, "SE_with_Hyena" ,"SE_Hyena",isPerturbed=True,perturbation_type='shift',intensity=0.1)
  
        #calculate_Lime(model_hyena, imagepath, DEVICE, dataset, "SE_with_Hyena")
       # counterfactual(model,imagepath, label)
       # counterfactual(model_hyena,imagepath, label)
    #caluclateShap(model,val_loader, dataset.categories, target_layer, listOfImages)
    #caluclateShap(model_hyena,val_loader, dataset.categories, hyena_target_layer, listOfImages)



def AA_experiments(dataset, train_loader, val_loader, listOfImages):
    AA = timm.create_model('seresnet33ts.ra2_in1k', pretrained=True, num_classes=NUM_CLASSES)
    pretrain_freeze_AA(AA)

    AA.to(DEVICE)
   
    
    model = get_model(AA, "AA", train_loader, val_loader, LOSS, LEARNING_RATE, WEIGHT_DECAY, EPOCH)

    AA_Hyena = timm.create_model('seresnet33ts.ra2_in1k', pretrained=True, num_classes=NUM_CLASSES)

    pretrain_freeze_AA(AA_Hyena,"hyena")

    AA_Hyena.to(DEVICE)
    model_hyena = get_model(AA_Hyena, "AA_with_Hyena", train_loader,val_loader, HYENA_LOSS, HYENA_LEARNING_RATE, HYENA_WEIGHT_DECAY, EPOCH)

    
    #validate(model, train_loader, LOSS)
    #validate(model, val_loader, LOSS)
    #validate(model_hyena, train_loader, HYENA_LOSS)
    #validate(model_hyena, val_loader, HYENA_LOSS)


    
    #plot_metrics("AA", "AA_with_Hyena", output_dir=OUTPUT_DIR)
   #get_median_time("AA", "AA_with_Hyena")
    
    #check_class_predictions_by_index(model, train_loader, NUM_CLASSES, DEVICE ,"AA", "training")
    #find_class_predictions_by_index(model, model_hyena, train_loader, NUM_CLASSES, DEVICE ,"AA","training")
    #find_class_predictions_by_index(model, model_hyena, val_loader, NUM_CLASSES, DEVICE ,"AA","validation")
    #check_class_predictions_by_index(model, val_loader, NUM_CLASSES, DEVICE, "AA", "validation")
   # check_class_predictions_by_index(model_hyena, train_loader, NUM_CLASSES, DEVICE, "AA_Hyena", "training")
    #check_class_predictions_by_index(model_hyena, val_loader, NUM_CLASSES, DEVICE, "AA_Hyena", "validation")
   
    
    target_layer = [model.stages[0][1].conv2, model.stages[1][1].conv2,model.stages[2][1].conv2,model.stages[3][1].conv2]#model.stages[0][1].attn, model.stages[1][1].attn,model.stages[2][1].attn,model.stages[3][1].attn]
    hyena_target_layer = [model_hyena.stages[0][1].conv2, model_hyena.stages[1][1].conv2,model_hyena.stages[2][1].conv2,model_hyena.stages[3][1].conv2]#model_hyena.stages[0][1].attn,model_hyena.stages[1][1].attn,model_hyena.stages[2][1].attn,model_hyena.stages[3][1].attn]
#   
    #print(model_hyena)
    for imagepath, label in listOfImages:
        
        pass
        
        calculate_Grad_CAM(model, target_layer, imagepath, "AA","AA")
        #calculate_Grad_CAM(model, target_layer, imagepath, "SE","SE", isPerturbed=True, perturbation_type='noise', intensity=0.04)
        #calculate_Grad_CAM(model, target_layer, imagepath, "SE","SE", isPerturbed=True, perturbation_type='blur', intensity=1.1)
        #calculate_Grad_CAM(model, target_layer, imagepath, "SE","SE", isPerturbed=True, perturbation_type='shift', intensity=0.1)
     
       # calculate_Lime(model, imagepath, DEVICE, dataset, "SE")

        calculate_Grad_CAM(model_hyena, hyena_target_layer, imagepath, "AA_with_Hyena" ,"AA_Hyena")
      #  calculate_Grad_CAM(model_hyena, hyena_target_layer, imagepath, "AA_with_Hyena" ,"AA_Hyena",isPerturbed=True,intensity=0.04)
    #    calculate_Grad_CAM(model_hyena, hyena_target_layer, imagepath, "AA_with_Hyena" ,"AA_Hyena",isPerturbed=True, perturbation_type='blur',intensity=1.1)
      #  calculate_Grad_CAM(model_hyena, hyena_target_layer, imagepath, "AA_with_Hyena" ,"AA_Hyena",isPerturbed=True,perturbation_type='shift',intensity=0.1)
  
        #calculate_Lime(model_hyena, imagepath, DEVICE, dataset, "SE_with_Hyena")
        #counterfactual(model,imagepath, label+1)
       # counterfactual(model_hyena,imagepath, label+1)
    #caluclateShap(model,val_loader, dataset.categories, target_layer, listOfImages)
    #caluclateShap(model_hyena,val_loader, dataset.categories, hyena_target_layer, listOfImages)
    
       

from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

def check_class_predictions_by_index(model, val_loader, num_classes, device, model_type, loaderType, save_correct_predictions="correct_predictions.txt"):
    """
    Check which classes the model predicts correctly in the validation dataset and save correctly predicted image names.

    Args:
        model: The trained model to evaluate.
        val_loader: DataLoader containing the validation dataset.
        num_classes: The total number of classes (e.g., 257 for Caltech-256).
        device: Device to run the model on (e.g., "cuda" or "cpu").
        model_type: The type of model being used.
        loaderType: The type of dataset loader.
        save_correct_predictions: Path to save correctly predicted image names.

    Returns:
        correct_predictions: A dictionary mapping each class index to a list of correctly predicted sample indices.
        incorrect_predictions: A dictionary mapping each class index to a list of incorrectly predicted sample indices.
    """
    model.eval()  # Set the model to evaluation mode

    # Initialize dictionaries to store correct and incorrect predictions for each class
    correct_predictions = defaultdict(list)
    incorrect_predictions = defaultdict(list)
    correct_image_filenames = []  # List to store correctly predicted image filenames

    # Loop through the validation dataset
    with torch.no_grad():  # No need to compute gradients during evaluation
        for i, (images, labels, paths) in enumerate(val_loader):  # Assuming paths are returned by dataset
            # Move data to the correct device
            images = images.to(device)
            labels = labels.to(device)

            # Get model predictions
            outputs = model(images)

            # Get the predicted class by taking the index of the maximum output (highest probability)
            _, predicted = torch.max(outputs, 1)

            # Loop through each image in the batch
            for idx in range(labels.size(0)):
                true_label = labels[idx].item()
                predicted_label = predicted[idx].item()

                # Check if the prediction is correct
                image_path = paths[idx]  # Extract filename from paths list
                if predicted_label == true_label:
                    correct_predictions[true_label].append(i * val_loader.batch_size + idx)
                    correct_image_filenames.append(image_path)
                else:
                    incorrect_predictions[true_label].append(i * val_loader.batch_size + idx)

    # Save correct image filenames to a file
    with open(save_correct_predictions, "w") as f:
        for filename in correct_image_filenames:
            f.write(filename + "\n")
    print(f"Saved correctly predicted image names to {save_correct_predictions}")

    # Compute per-class accuracy
    correct_counts = np.zeros(num_classes)
    incorrect_counts = np.zeros(num_classes)

    for class_idx in range(num_classes):
        correct_counts[class_idx] = len(correct_predictions[class_idx])
        incorrect_counts[class_idx] = len(incorrect_predictions[class_idx])

    total_counts = correct_counts + incorrect_counts
    accuracy_per_class = np.divide(correct_counts, total_counts, out=np.zeros_like(correct_counts), where=total_counts != 0)

    # Get the top 10 classes with the highest accuracy
    top_10_indices = np.argsort(accuracy_per_class)[-10:][::-1]
    print(f"Top 10 Classes with Highest Correct Prediction Rate {model_type} for {loaderType}:")
    for idx in top_10_indices:
        print(f"Class Index: {idx}, Accuracy: {accuracy_per_class[idx]:.2f}, Correct Predictions: {int(correct_counts[idx])}, Total Predictions: {int(total_counts[idx])}")
    print('---------------------------------------')

    # Plot correct predictions
    x = np.arange(num_classes)  # Class indices for the x-axis
    bar_width = 0.6  # Reduced bar width for more space between bars
    tick_spacing = 10

    fig, ax1 = plt.subplots(figsize=(20, 8))  # Increased figure size for readability
    ax1.bar(x, correct_counts, width=bar_width, color='green', label='Correct Predictions')

    ax1.set_xlabel('Class Index')
    ax1.set_ylabel('Number of Correct Predictions')
    ax1.set_title(f'{model_type} Correct Predictions Per Class for {loaderType}')
    ax1.set_xticks(x[::tick_spacing])  # Show fewer x-axis ticks
    plt.xticks(rotation=90)
    ax1.grid(axis='y')  # Add horizontal gridlines
    plt.tight_layout()
    plt.show()

    # Plot incorrect predictions
    fig, ax2 = plt.subplots(figsize=(20, 8))  
    ax2.bar(x, incorrect_counts, width=bar_width, color='red', label='Incorrect Predictions')

    ax2.set_xlabel('Class Index')
    ax2.set_ylabel('Number of Incorrect Predictions')
    ax2.set_title(f'{model_type} Incorrect Predictions Per Class {loaderType}')
    ax2.set_xticks(x[::tick_spacing])
    plt.xticks(rotation=90)
    ax2.grid(axis='y')  
    plt.tight_layout()
    plt.show()

    return correct_predictions, incorrect_predictions



def find_class_predictions_by_index(model,model2, val_loader, num_classes, device, model_type, loaderType):
    model.eval()  
    model2.eval()

    correct_predictions = defaultdict(list)
    incorrect_predictions = defaultdict(list)

    # Retrieve original dataset
    original_dataset = val_loader.dataset.dataset  # This is the full Caltech256 dataset

    # File to save correctly predicted image names
    correct_file = f"correct_predictions_{model_type}_{loaderType}.txt"

    with torch.no_grad():
        with open(correct_file, "w") as f:  # Open file to save correct image names
            for i, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                outputs2 = model2(images)
                _, predicted = torch.max(outputs, 1)
                _, predicted2 = torch.max(outputs2, 1)
                

                batch_start_idx = i * val_loader.batch_size  # Batch start index

                for idx in range(labels.size(0)):
                    true_label = labels[idx].item()
                    predicted_label = predicted[idx].item()
                    predicted_label2 = predicted2[idx].item()

                    # Get the dataset index
                    dataset_index = val_loader.dataset.indices[batch_start_idx + idx]

                    # Reconstruct file path
                    category = original_dataset.categories[original_dataset.y[dataset_index]]
                    image_number = original_dataset.index[dataset_index]
                    file_name = f"{original_dataset.y[dataset_index] + 1:03d}_{image_number:04d}.jpg"
                    file_path = os.path.join(original_dataset.root, "256_ObjectCategories", category, file_name)

                    # Save file name if prediction is correct
                    if predicted_label == true_label and predicted_label2 == true_label:
                        correct_predictions[true_label].append(dataset_index)
                        print(file_name)
                        print(f"Predictions: Model1={predicted_label}, Model2={predicted_label2}, True={true_label}")
                        f.write(f"{file_name}\n")  # Save filename
                    else:
                        incorrect_predictions[true_label].append(dataset_index)

    print(f"Correctly classified image names saved to {correct_file}")

    return correct_predictions, incorrect_predictions

def test_SE(dataset, train_loader, val_loader):
    SE = SEResNet50(SEBasicBlock,NUM_CLASSES,False, dropout_rate=0.5).to(DEVICE)
    #init_weights(SE)
    optim = torch.optim.SGD(SE.parameters(), lr=1e-7, weight_decay=1e-2, momentum=0.9)
    lr_finder = LearningRateFinder(SE, LOSS, optim, train_loader)
    lr_finder.find_lr(init_lr=1e-7, final_lr=1e-1, num_iter=100)
    lr_finder.plot_lr_finder()



def measure_vram_usage(model, input_tensor):
    # Clear cache
    torch.cuda.empty_cache()
    
    # Get initial memory
    initial_mem = torch.cuda.memory_allocated()
    
    # Forward pass
    output = model(input_tensor)
    
    # Get peak memory during forward pass
    peak_mem = torch.cuda.max_memory_allocated()
    
    # Calculate memory used
    memory_used = peak_mem - initial_mem
    
    # Convert to MB for readability
    return memory_used / (1024 ** 2)  # in MB