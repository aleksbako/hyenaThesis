import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from PIL import Image
import os

class ImageNetValDataset(Dataset):
    def __init__(self, val_dir, ground_truth_file, id_to_wnid, transform=None):
        self.val_dir = val_dir
        self.transform = transform
        self.id_to_wnid = id_to_wnid
        self.image_paths = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith('.JPEG')]
        

        with open(ground_truth_file, 'r') as f:
            self.labels = [int(line.strip()) for line in f]  # Read as 1-based indices

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        ilsvrc_id = self.labels[idx]  # 1-based index

        # Map ILSVRC ID to WNID (folder name)
        wnid = self.id_to_wnid[str(ilsvrc_id)]

        if self.transform:
            image = self.transform(image)

        return image, ilsvrc_id
