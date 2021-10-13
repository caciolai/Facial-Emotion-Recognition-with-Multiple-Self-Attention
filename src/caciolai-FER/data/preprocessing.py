from typing import Tuple
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data.dataloader import DataLoader

from .datasets import BalancedSampler, MyDataset
from .augmentation import get_augmentation_transforms


BATCH_SIZE = 128


def clean_data(fer: pd.DataFrame, ferplus: pd.DataFrame) -> pd.DataFrame:
    """Corrects original FER dataset with improved labels from FER+

    :param fer: the original FER dataset
    :type fer: pd.DataFrame
    :param ferplus: the improved FER+ dataset (labels only)
    :type ferplus: pd.DataFrame
    :return: Corrected FER dataset
    :rtype: pd.DataFrame
    """
    # drop usage and emotion in fer
    fer = fer.drop(["Usage"], axis=1)
    # concatenate
    df = pd.concat([fer, ferplus], axis=1)
    
    # keep ferplus labels
    df["label"] = df[["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt", "unknown", "NF"]].idxmax(axis=1)
    df = df[["pixels", "Usage", "label"]]

    # get rid of ambiguous faces
    df = df.drop(df[df["label"] == "NF"].index)

    # get rid of unknown/rare emotion
    df = df.drop(df[df["label"] == "contempt"].index)
    df = df.drop(df[df["label"] == "unknown"].index)

    df.reset_index(inplace=True, drop=True)

    return df

def prepare_dataset(data: pd.DataFrame, label_encoder: LabelEncoder) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepares a dataset for PyTorch

    :param data: raw dataset
    :type data: pd.DataFrame
    :param label_encoder: label encoder
    :type label_encoder: LabelEncoder
    :return: X, y feature and label tensors
    :rtype: Tuple[torch.Tensor, torch.Tensor]
    """
    width = 48
    height = 48
    X = np.zeros((len(data), height, width), dtype=np.uint8)

    y = label_encoder.transform(data["label"].to_numpy())
    y = torch.Tensor(y).to(torch.long)

    for i, row in enumerate(data.index):
        pixels = np.fromstring(data['pixels'][row], dtype=int, sep=' ')
        image = np.asarray(pixels).reshape(48, 48)
        image = image.astype(np.uint8)
        X[i] = image
        # X[i] = np.expand_dims(image, -1)

    return X, y

def prepare_datasets(df: pd.DataFrame) -> Tuple[MyDataset, MyDataset, MyDataset]:
    """Prepare all datasets (they come already sorted in train, val, test sets)

    :param df: Cleaned FER dataframe
    :type df: pd.DataFrame
    :return: The three torch (X, y) datasets
    :rtype: Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
    """
    le = LabelEncoder()
    le.fit(df["label"].to_numpy())
    
    X_train, y_train    = prepare_dataset(df[df['Usage'] == 'Training'], le)
    X_val, y_val        = prepare_dataset(df[df['Usage'] == 'PrivateTest'], le)
    X_test, y_test      = prepare_dataset(df[df['Usage'] == 'PublicTest'], le)

    train_transform, test_transform = get_augmentation_transforms()

    train_set   = MyDataset(X_train, y_train, train_transform)
    val_set     = MyDataset(X_val, y_val, test_transform)
    test_set    = MyDataset(X_test, y_test, test_transform)

    return train_set, val_set, test_set


def prepare_dataloaders(df: pd.DataFrame) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Prepares all dataloaders for training and evaluation

    :param df: Cleaned FER dataframe
    :type df: pd.DataFrame
    :return: [description]
    :rtype: Tuple[DataLoader, DataLoader, DataLoader]
    """
    train_set, val_set, test_set = prepare_dataset()

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, sampler=BalancedSampler(train_set))
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, sampler=BalancedSampler(val_set))
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader

