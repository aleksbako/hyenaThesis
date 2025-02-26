import scipy.io
from .dataset.ImageNetValDataset import ImageNetValDataset
from torch.utils.data import DataLoader , random_split
import torch
from torchvision import datasets
import torchvision
from util import IMAGE_SIZE, BATCH_SIZE, LOSS, EPOCH, MEAN, STD, NUM_CLASSES
def getCaltechDataLoaders(train_transform, val_transform):
    cal_dataset = torchvision.datasets.Caltech256(root="../data/",download=False)
        # Calculate the sizes for the splits
    train_size = int(0.85 * len(cal_dataset))
    val_size = len(cal_dataset) - train_size #int(0.2 * len(cal_dataset))
   # test_size = len(cal_dataset) - train_size - val_size  # Remaining data for testing

    train_dataset, val_dataset = random_split(cal_dataset, [train_size, val_size], torch.Generator().manual_seed(11))

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    #test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Apply the respective transformations
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    return train_loader, val_loader, cal_dataset