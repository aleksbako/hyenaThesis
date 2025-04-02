import torch
import torchvision.transforms as transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#LEARNING_RATE=1e-3
#LEARNING_RATE=1e-4 #SE
#LEARNING_RATE=1e-3 #VIT
#WEIGHT_DECAY=1e-4
LEARNING_RATE=1e-3 #VIT
WEIGHT_DECAY=1e-4
LOSS=torch.nn.CrossEntropyLoss(label_smoothing=0.1)
OPTIMIZER_TYPE="ADAM" #SGD for SE, AA
#HYENA_LEARNING_RATE=3e-4 #SE
HYENA_LEARNING_RATE=1e-3 #VIT
#HYENA_WEIGHT_DECAY=2e-5 #VIT
HYENA_WEIGHT_DECAY=1e-4 #SE
HYENA_LOSS=torch.nn.CrossEntropyLoss(label_smoothing=0.1)
HYENA_OPTIMIZER_TYPE="ADAM" #SGD for SE, MOBILE, AA
DATA_DIR = '../data/'
OUTPUT_DIR = '../output/'

NUM_CLASSES = 257#257
IMAGE_SIZE = 128#224#64 #224 #128 #64 #224
BATCH_SIZE=32#16
#EPOCH=30
EPOCH=80 #250#80#263 #250 #60 SE
MEAN=[0.485, 0.456, 0.406]
STD=[0.229, 0.224, 0.225]
VARMUP_EPOCH=15#15#40
#VARMUP_EPOCH=15 #SE
VAL_TRANSFORMATION = transforms.Compose([
    transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    #transforms.Grayscale(num_output_channels=1),  # Keep grayscale but single channel
    transforms.Lambda(lambda x: x.convert("RGB")),
    #transforms.Grayscale(num_output_channels=3),  # Resize the image to 256x256
    #transforms.CenterCrop((IMAGE_SIZE,IMAGE_SIZE)),  # Crop the center of the image to 224x224                # Resize to the same size as training set
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)                   
])
TRAIN_TRANSFORMATION = transforms.Compose([
    transforms.Resize(IMAGE_SIZE+32),
    transforms.RandomCrop((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    #ransforms.Grayscale(num_output_channels=3),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    #transforms.GaussianBlur(kernel_size=5),
    transforms.RandomErasing(p=0.3), #SE 0.5
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
    ]
    )

