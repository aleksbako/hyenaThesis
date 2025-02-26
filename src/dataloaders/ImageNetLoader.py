import scipy.io
from .dataset.ImageNetValDataset import ImageNetValDataset
from torch.utils.data import DataLoader , random_split
from torchvision import datasets
from util import IMAGE_SIZE, BATCH_SIZE, LOSS, EPOCH, MEAN, STD, NUM_CLASSES
def getImageNetDataLoaders(train_transform, val_transform):

    # Load the .mat file
    mat_data = scipy.io.loadmat('../data/meta.mat')
    # Extract the synsets array
    synsets = mat_data['synsets']
    import json
    # Prepare mappings
    id_to_wnid = {}
    wnid_to_id = {}
    # Extract data
    for synset in synsets:
        ilsvrc_id = int(synset['ILSVRC2012_ID'][0][0])
        wnid = synset['WNID'][0][0]
        id_to_wnid[ilsvrc_id] = wnid
        wnid_to_id[wnid] = ilsvrc_id
    # Save to JSON for easy access in Python
    with open('synset_mapping.json', 'w') as f:
        json.dump(id_to_wnid, f, indent=4)
    with open('synset_mapping.json') as f:
        id_to_wnid = json.load(f)
   # Load the datasets
    train_dataset = datasets.ImageFolder(root='../data/train')
    val_dataset = ImageNetValDataset('../data/val', "../data/ILSVRC2012_validation_ground_truth.txt",id_to_wnid, transform=val_transform)
    train_dataset.transform = train_transform
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    return train_loader , val_loader, train_dataset