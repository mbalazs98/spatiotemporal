from utils import set_seed

import numpy as np

from torch.utils.data import DataLoader, Dataset
import torch

import tonic
import tonic.transforms as transforms

        
def RawHD_dataloaders(config):
    set_seed(config.seed)

    train_dataset = RawHD(config.datasets_path, train=True)
    test_dataset = RawHD(config.datasets_path, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=4)
    
    return train_loader, test_loader

class RawHD(Dataset):
    def __init__(self, data_path, train=True):
        if train:
            self.x_data = np.load(data_path+"/training_x_data.npy") * 20#12.5
            self.y_data = np.load(data_path+"/training_y_data.npy")
        else:
            self.x_data = np.load(data_path+"/testing_x_data.npy") * 20#12.5
            self.y_data = np.load(data_path+"/testing_y_data.npy")

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = self.x_data[idx]
        y = self.y_data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y)
    

def DVSGesture_dataloaders(config):
    set_seed(config.seed)
    sensor_size = tonic.datasets.DVSGesture.sensor_size
    frame_transform = transforms.ToFrame(sensor_size=sensor_size, n_time_bins=80)
    transform = transforms.Compose([frame_transform])
    train_dataset = tonic.datasets.DVSGesture(save_to=config.datasets_path, train=True, transform=transform)
    test_dataset = tonic.datasets.DVSGesture(save_to=config.datasets_path, train=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=4)
    
    return train_loader, test_loader
