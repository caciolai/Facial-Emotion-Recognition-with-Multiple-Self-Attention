from typing import *

# PyTorch training framework
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data.dataloader import DataLoader

from sklearn.preprocessing import LabelEncoder

# Metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Plotting utilities
from matplotlib import pyplot as plt
import seaborn as sns


def test(
    model: pl.LightningModule,
    test_loader: DataLoader,
):

    tester = pl.Trainer(gpus=-1)
    tester.test(model, test_loader)

def compute_predictions(
    model: pl.LightningModule,
    test_loader: DataLoader,
) -> List[List[int]]:
    """Compute predictions from a trained model over test data.

    :param model: trained model
    :type model: pl.LightningModule
    :param test_loader: dataloader with test data
    :type test_loader: DataLoader
    :return: predictions (batched)
    :rtype: List[List[int]]
    """
    
    tester = pl.Trainer(gpus=-1)
    y_preds = tester.predict(model, test_loader)
    
    return y_preds

def compute_classification_report(
    model: pl.LightningModule,
    test_loader: DataLoader,
    label_encoder: LabelEncoder = None
):

    y_preds = compute_predictions(model, test_loader)

    y_true = list()
    y_pred = list()

    for i,batch in enumerate(test_loader):
        inputs, labels = batch

        y_true += labels.tolist()
        y_pred += y_preds[i].tolist()


    target_names = label_encoder.classes_ if label_encoder is not None else None
    print(classification_report(y_true, y_pred, target_names=target_names))


def plot_confusion_matrix(
    model: pl.LightningModule,
    test_loader: DataLoader,
    label_encoder: LabelEncoder = None
):

    y_preds = compute_predictions(model, test_loader)

    y_true = list()
    y_pred = list()

    for i,batch in enumerate(test_loader):
        inputs, labels = batch

        y_true += labels.tolist()
        y_pred += y_preds[i].tolist()

    target_names = label_encoder.classes_ if label_encoder is not None else None

    cm = confusion_matrix(y_true, y_pred, normalize='true')

    # df_cm = pd.DataFrame(cm, target_names, target_names)
    # plt.figure(figsize=(10,7))
    # sns.set(font_scale=1.4) # for label size
    # sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size

    fig, ax = plt.subplots(figsize=(10,10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)

    disp.plot(ax=ax, cmap=plt.cm.Blues)

    fig.show()

