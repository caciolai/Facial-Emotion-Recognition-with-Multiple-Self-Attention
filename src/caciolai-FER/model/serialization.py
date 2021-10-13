from typing import *

import torch
import pytorch_lightning as pl
import os
import pathlib
from datetime import datetime


STATE_DICT_DIR = "./models"


def save_model(model: pl.LightningModule):
    """Saves the current model weight dir on disk

    :param model: model
    :type model: pl.LightningModule
    """
    state_dict_dir = STATE_DICT_DIR
    
    if not os.path.isdir(state_dict_dir):
        pathlib.Path(state_dict_dir).mkdir(parents=True, exist_ok=True)

    state_dict_name = f"{model.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pt"

    state_dict_path = os.path.join(state_dict_dir, state_dict_name)
    print("Saving model at:", state_dict_path)
    torch.save(model.state_dict(), state_dict_path)

