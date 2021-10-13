from typing import Tuple
import torchvision.transforms as transforms
import numpy as np
from torchvision.transforms.transforms import Compose


def get_train_transform() -> transforms.Compose:
    """Data augmentation pipeline for training consists in:
        - cast to a PIL object (better handling by torchvision)
        - random horizontal flipping (50% chance)
        - random rotation in the range of [0, pi/5]
        - cast back to tensor object

    :return: Data augmentation pipeline for training
    :rtype: transforms.Compose
    """
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(0.2*np.pi),
        # transforms.RandomPerspective(distortion_scale=0.2),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,),(0.5,))
    ])
    return train_transform

def get_test_transform() -> transforms.Compose:
    """Transform consists in same steps of train transform without augmentation

    :return: Data transform pipeline for evaluation
    :rtype: transforms.Compose
    """
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,),(0.5,))
    ])
    return test_transform

def get_augmentation_transforms() -> Tuple[Compose, Compose]:
    """Get transforms for training and evaluation

    :return: Train and test transforms
    :rtype: Tuple[Compose, Compose]
    """
    train_transform = get_train_transform()
    test_transform = get_test_transform()

    return train_transform, test_transform