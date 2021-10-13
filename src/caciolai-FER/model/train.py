from typing import *

# PyTorch training framework
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data.dataloader import DataLoader


def get_callbacks() -> List[pl.Callback]:
    num_epochs = 300

    tb_logger = pl_loggers.TensorBoardLogger('./logs/')

    early_stop_callback = EarlyStopping(
        monitor='val_acc',
        min_delta=0.00,
        patience=num_epochs/10,
        verbose=False,
        mode='max'
    )

    return [tb_logger, early_stop_callback]


def train(
    model: pl.LightningModule,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 300,
):

    callbacks = get_callbacks()
    trainer = pl.Trainer(max_epochs=num_epochs, gpus=-1, callbacks=callbacks)
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)

