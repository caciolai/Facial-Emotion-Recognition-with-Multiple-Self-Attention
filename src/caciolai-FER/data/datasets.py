import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler

class MyDataset(Dataset):
    def __init__(self, images, labels, transform=None, augment=False):
        self.images = images
        self.labels = labels
        self.transform = transform

        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        sample = (img, label)

        return sample

class BalancedSampler(WeightedRandomSampler):
    def __init__(self, dataset):
        y = dataset.labels
        
        class_sample_count = np.array([len(np.where(y==t)[0]) for t in np.unique(y)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in y])

        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.to(torch.double)
        
        # sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        super().__init__(samples_weight, len(samples_weight))
