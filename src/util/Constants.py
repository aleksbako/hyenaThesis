import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE=2e-4
WEIGHT_DECAY=5e-2
LOSS=torch.nn.CrossEntropyLoss()

HYENA_LEARNING_RATE=1e-5
HYENA_WEIGHT_DECAY=5e-2
HYENA_LOSS=torch.nn.CrossEntropyLoss()

DATA_DIR = '../data/'
OUTPUT_DIR = '../output/'

NUM_CLASSES = 257
IMAGE_SIZE = 224
BATCH_SIZE=100
EPOCH=50
MEAN=[0.4914, 0.4822, 0.4465]
STD=[0.2023, 0.1994, 0.2010]
