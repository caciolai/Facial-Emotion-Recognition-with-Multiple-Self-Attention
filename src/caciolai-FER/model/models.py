# DL
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.layers import SelfAttention

# PyTorch training framework
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

from .layers import SpatialTransformer


class MyLightningModule(pl.LightningModule):
    def __init__(self, Ncrops=False):
        super().__init__()
        self.Ncrops = Ncrops

    def training_step(self, batch, batch_idx):
        inputs, labels = batch

        if self.Ncrops:
            # fuse crops and batchsize
            bs, ncrops, c, h, w = inputs.shape
            inputs = inputs.view(-1, c, h, w)
            # repeat labels ncrops times
            labels = torch.repeat_interleave(labels, repeats=ncrops, dim=0)

        logits = self(inputs)
        loss = F.cross_entropy(logits, labels)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch

        if self.Ncrops:
            # fuse crops and batchsize
            bs, ncrops, c, h, w = inputs.shape
            inputs = inputs.view(-1, c, h, w)
            # forward
            logits = self(inputs)
            # combine results across the crops
            logits = logits.view(bs, ncrops, -1)
            logits = torch.sum(logits, dim=1) / ncrops
        else:
            logits = self(inputs)

        loss = F.cross_entropy(logits, labels)
        probs = F.log_softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        acc = accuracy(preds, labels)        
        
        metrics = {'val_acc': acc, 'val_loss': loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        inputs, labels = batch

        if self.Ncrops:
            # fuse crops and batchsize
            bs, ncrops, c, h, w = inputs.shape
            inputs = inputs.view(-1, c, h, w)
            # forward
            logits = self(inputs)
            # combine results across the crops
            logits = logits.view(bs, ncrops, -1)
            logits = torch.sum(logits, dim=1) / ncrops
        else:
            logits = self(inputs)

        loss = F.cross_entropy(logits, labels)
        probs = F.log_softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        acc = accuracy(preds, labels)     

        metrics = {'test_acc': acc, 'test_loss': loss}
        self.log_dict(metrics)
        return metrics

    def predict_step(self, batch, batch_idx, dataloader_idx):
        inputs, _ = batch
        
        logits = self(inputs)
        probs = F.log_softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        return preds

class SimpleCNN(MyLightningModule):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1a = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3)
        self.conv1b = nn.Conv2d(10, out_channels=10, kernel_size=3)

        self.conv2a = nn.Conv2d(10, 10, 3)
        self.conv2b = nn.Conv2d(10, 10, 3)

        self.lin1 = nn.Linear(10 * 9 * 9, 50)
        self.lin2 = nn.Linear(50, num_classes)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop = nn.Dropout()

    def forward(self, x):
        # (1, 48, 48) -> (10, 46, 46)
        x = F.relu(self.conv1a(x))
        # (10, 46, 46) -> (10, 44, 44)
        x = F.relu(self.conv1b(x))
        # (10, 44, 44) -> (10, 22, 22)
        x = self.pool(x)

        # (10, 22, 22) -> (10, 20, 20)
        x = F.relu(self.conv2a(x))
        # (10, 20, 20) -> (10, 18, 18)
        x = F.relu(self.conv2b(x))
        # (10, 18, 18) -> (10, 9, 9)
        x = self.pool(x)
        x = self.drop(x)

        # (10, 9, 9) -> (10 * 9 * 9,)
        x = x.view(-1, 10 * 9 * 9)
        # (10 * 9 * 9,) -> (50,)
        x = F.relu(self.lin1(x))
        # (50,) -> (num_classes,)
        x = self.lin2(x)

        return x
    
    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        optimizer = torch.optim.SGD(
            self.parameters(), lr=0.01, momentum=0.9, nesterov=True, 
            weight_decay=0.0001)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=False)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_acc'
        }

class DeepEmotion(MyLightningModule):
    def __init__(self, num_classes):
        '''
        https://github.com/omarsayed7/Deep-Emotion/blob/master/deep_emotion.py
        '''
        super().__init__()
        self.conv1 = nn.Conv2d(1,10,3)
        
        self.conv2 = nn.Conv2d(10,10,3)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(10,10,3)
        self.conv4 = nn.Conv2d(10,10,3)
        self.pool4 = nn.MaxPool2d(2,2)

        self.norm = nn.BatchNorm2d(10)

        self.fc1 = nn.Linear(810,50)
        self.fc2 = nn.Linear(50,num_classes)

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(640, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 640)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x, grid

    def forward(self, x):
        x, _ = self.stn(x)

        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = F.relu(self.pool2(x))

        x = F.relu(self.conv3(x))
        x = self.norm(self.conv4(x))
        x = F.relu(self.pool4(x))

        # out = F.dropout(out)
        x = x.view(-1, 810)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        optimizer = torch.optim.SGD(
            self.parameters(), lr=0.01, momentum=0.9, nesterov=True, 
            weight_decay=0.0001)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=False)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_acc'
        }

class MultipleSelfAttention(nn.Module):    
    def __init__(self, shape, num_heads):
        super().__init__()

        self.ch, self.h, self.w = shape
        self.n = num_heads

        n, ch, w, h = self.n, self.ch, self.w, self.h
        # hybrid attention
        self.attention = SelfAttention(self.n*self.ch)
        
        # weight learning
        self.conv = nn.Conv2d(in_channels=ch, out_channels=ch, kernel_size=1, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        h, w = h//2, w//2
        self.fc = nn.Linear(ch*h*w, num_heads)

    def forward(self, x):
        n, ch, w, h = self.n, self.ch, self.w, self.h
        
        # hybrid attention
        xh = x.repeat(1, n, 1, 1)
        xh = self.attention(xh)
        xh = xh.view(-1, n, ch, h, w) # (bs, N, ch, h, w)
        
        # weight learning
        # (bs, ch, h, w) -> (bs, ch, h, w)
        xs = self.conv(x) 
        # (bs, ch, h, w) -> (bs, ch, h/2, w/2)
        xs = self.pool(x) 
        # (bs, ch, h/2, w/2) -> (bs, ch*h/2*w/2)
        h, w = h//2, w//2
        xs = xs.view(-1, ch*h*w)
        # (bs, ch*h/2*w/2) -> (bs, n)
        xs = F.sigmoid(self.fc(xs))
        xs = F.normalize(xs, p=1, dim=1) # obtain probabilities

        # weighted sum
        x = torch.sum(torch.mul(xh, xs[:,:,None,None,None]), dim=1)

        return x

class DeepEmotionAttention(MyLightningModule):
    def __init__(self, num_classes):
        '''
        https://github.com/omarsayed7/Deep-Emotion/blob/master/deep_emotion.py
        '''
        super().__init__()
        self.conv1 = nn.Conv2d(1,10,3)
        
        self.conv2 = nn.Conv2d(10,10,3)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv3 = nn.Conv2d(10,10,3)
        self.conv4 = nn.Conv2d(10,10,3)
        self.pool4 = nn.MaxPool2d(2,2)

        self.norm = nn.BatchNorm2d(10)

        self.fc1 = nn.Linear(810,50)
        self.fc2 = nn.Linear(50,num_classes)

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            MultipleSelfAttention((8, 42, 42), 8),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(640, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 640)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x, grid

    def forward(self, x):
        x, _ = self.stn(x)

        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = F.relu(self.pool2(x))

        x = F.relu(self.conv3(x))
        x = self.norm(self.conv4(x))
        x = F.relu(self.pool4(x))

        # out = F.dropout(out)
        x = x.view(-1, 810)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        optimizer = torch.optim.SGD(
            self.parameters(), lr=0.01, momentum=0.9, nesterov=True, 
            weight_decay=0.0001)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=False)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_acc'
        }

class VGGFaceAttention(MyLightningModule):
    def __init__(self, num_classes, Ncrops=False):
        super().__init__(Ncrops)

        self.conv1a = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding='same')
        self.conv1b = nn.Conv2d(64, out_channels=64, kernel_size=3, padding='same')

        self.msa = MultipleSelfAttention((64, 24, 24), 8) # combine N attention heads

        self.conv2a = nn.Conv2d(64, 128, 3, padding='same')
        self.conv2b = nn.Conv2d(128, 128, 3, padding='same')

        self.conv3a = nn.Conv2d(128, 256, 3, padding='same')
        self.conv3b = nn.Conv2d(256, 256, 3, padding='same')

        self.conv4a = nn.Conv2d(256, 512, 3, padding='same')
        self.conv4b = nn.Conv2d(512, 512, 3, padding='same')

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn1a = nn.BatchNorm2d(64)
        self.bn1b = nn.BatchNorm2d(64)

        self.bn2a = nn.BatchNorm2d(128)
        self.bn2b = nn.BatchNorm2d(128)

        self.bn3a = nn.BatchNorm2d(256)
        self.bn3b = nn.BatchNorm2d(256)

        self.bn4a = nn.BatchNorm2d(512)
        self.bn4b = nn.BatchNorm2d(512)

        self.lin1 = nn.Linear(512 * 3 * 3, 4096)
        self.lin2 = nn.Linear(4096, 4096)
        self.lin3 = nn.Linear(4096, num_classes)

        self.drop = nn.Dropout()

    def forward(self, x):
        # (1, 48, 48) -> (64, 24, 24)
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool(x)

        # (64, 24, 24) -> (64, 24, 24)
        x = self.msa(x) # apply multiple self attention

        # (64, 24, 24) -> (128, 12, 12)
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool(x)

        # (128, 12, 12) -> (256, 6, 6)
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.pool(x)

        # (256, 6, 6) -> (512, 3, 3)
        x = F.relu(self.bn4a(self.conv4a(x)))
        x = F.relu(self.bn4b(self.conv4b(x)))
        x = self.pool(x)

        x = x.view(-1, 512 * 3 * 3)
        x = F.relu(self.drop(self.lin1(x)))
        x = F.relu(self.drop(self.lin2(x)))
        x = self.lin3(x)

        return x

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        optimizer = torch.optim.SGD(
            self.parameters(), lr=0.01, momentum=0.9, nesterov=True, 
            weight_decay=0.0001)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=False)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_acc'
        }

class VGGFaceAttentionSTN(MyLightningModule):
    def __init__(self, num_classes, Ncrops=False):
        super().__init__(Ncrops)

        
        self.conv1a = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding='same')
        self.conv1b = nn.Conv2d(64, out_channels=64, kernel_size=3, padding='same')
        
        self.stn = SpatialTransformer()
        self.msa = MultipleSelfAttention((64, 24, 24), 8) # combine N attention heads

        self.conv2a = nn.Conv2d(64, 128, 3, padding='same')
        self.conv2b = nn.Conv2d(128, 128, 3, padding='same')

        self.conv3a = nn.Conv2d(128, 256, 3, padding='same')
        self.conv3b = nn.Conv2d(256, 256, 3, padding='same')

        self.conv4a = nn.Conv2d(256, 512, 3, padding='same')
        self.conv4b = nn.Conv2d(512, 512, 3, padding='same')

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bn1a = nn.BatchNorm2d(64)
        self.bn1b = nn.BatchNorm2d(64)

        self.bn2a = nn.BatchNorm2d(128)
        self.bn2b = nn.BatchNorm2d(128)

        self.bn3a = nn.BatchNorm2d(256)
        self.bn3b = nn.BatchNorm2d(256)

        self.bn4a = nn.BatchNorm2d(512)
        self.bn4b = nn.BatchNorm2d(512)

        self.lin1 = nn.Linear(512 * 3 * 3, 4096)
        self.lin2 = nn.Linear(4096, 4096)
        self.lin3 = nn.Linear(4096, num_classes)

        self.drop = nn.Dropout()

    def forward(self, x):
        # (1, 48, 48) -> (64, 24, 24)
        x = F.relu(self.bn1a(self.conv1a(x)))
        x = F.relu(self.bn1b(self.conv1b(x)))
        x = self.pool(x)

        # (64, 24, 24) -> (64, 24, 24)
        x, _ = self.stn(x)
        x = self.msa(x) # apply multiple self attention

        # (64, 24, 24) -> (128, 12, 12)
        x = F.relu(self.bn2a(self.conv2a(x)))
        x = F.relu(self.bn2b(self.conv2b(x)))
        x = self.pool(x)

        # (128, 12, 12) -> (256, 6, 6)
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))
        x = self.pool(x)

        # (256, 6, 6) -> (512, 3, 3)
        x = F.relu(self.bn4a(self.conv4a(x)))
        x = F.relu(self.bn4b(self.conv4b(x)))
        x = self.pool(x)

        x = x.view(-1, 512 * 3 * 3)
        x = F.relu(self.drop(self.lin1(x)))
        x = F.relu(self.drop(self.lin2(x)))
        x = self.lin3(x)

        return x

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        optimizer = torch.optim.SGD(
            self.parameters(), lr=0.01, momentum=0.9, nesterov=True, 
            weight_decay=0.0001)
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=False)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_acc'
        }


