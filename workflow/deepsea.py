import numpy as np
import pandas as pd
import h5py
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

class BaseModel(pl.LightningModule):

    def __init__(self, lr=1e-4, momentum=0.9):
        super().__init__()
        self.lr = lr
        self.momentum = momentum

        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()
        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.val_acc = torchmetrics.Accuracy(task='binary')

    def _loss(self, batch):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        pred, loss = self._loss(batch)
        _, y = batch
        self.train_loss(loss.item())
        self.train_acc(pred, y)
        self.log('train_loss', self.train_loss, on_step=False, on_epoch=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred, loss = self._loss(batch)
        _, y = batch
        self.val_loss(loss.item())
        self.val_acc(pred, y)
        self.log('val_loss', self.val_loss, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            *self.parameters(),
        ], lr=self.lr)
        return optimizer


class ConvDNA(nn.Module):

    def __init__(self, hidden_size=500, dropout=.5, return_indices=False):
        super().__init__()

        self.conv1 = nn.Conv1d(4, 80, 8)
        self.conv2 = nn.Conv1d(80, 120, 8)
        self.conv3 = nn.Conv1d(120, 180, 8)
        self.pool = nn.MaxPool1d(kernel_size=4, stride=4,
                                 return_indices=return_indices)
        self.drop1 = nn.Dropout(p=dropout)
        self.drop2 = nn.Dropout(p=dropout)
        self.linear = nn.Linear(53*180, hidden_size)

    def forward(self, input):
        x = self.drop1(self.pool(F.relu(self.conv1(input))))
        x = self.drop1(self.pool(F.relu(self.conv2(x))))
        x = self.drop2(F.relu(self.conv3(x)))
        return self.linear(x.view(-1, 53*180))


class DeepSea(BaseModel):

    def __init__(self, output_size, dropout=.5, dna_hidden_size=100, lr=1e-3, momentum=0.9):
        super().__init__(lr, momentum)
        self.conv_dna = ConvDNA(dna_hidden_size, dropout=dropout)
        self.linear = nn.Linear(dna_hidden_size, output_size)

    def _forward(self, X):
        return self.linear(F.relu(self.conv_dna(X)))

    def forward(self, X):
        return F.sigmoid(self._forward(X))

    def _loss(self, batch):
        X, y = batch
        X = X.to(self.conv_dna.conv1.weight.dtype)
        y = y.to(self.linear.weight.dtype)
        pred = self._forward(X)
        return pred, F.binary_cross_entropy_with_logits(pred, y)

    def predict_step(self, batch, batch_idx):
        X, _ = batch
        X = X.to(self.conv_dna.conv1.weight.dtype)
        return self(X)


class DeepSeaGradCam(nn.Module):

    def __init__(self, output_size, dna_hidden_size=100):
        super().__init__()
        self.conv_dna = ConvDNA(dna_hidden_size, dropout=0.,
                                return_indices=True)
        self.linear = nn.Linear(dna_hidden_size, output_size)

    def forward(self, X):

        x = F.relu(self.conv_dna.conv1(X))
        a1 = x
        x, indices_pool1 = self.conv_dna.pool(x)

        x = F.relu(self.conv_dna.conv2(x))
        a2 = x
        x, indices_pool2 = self.conv_dna.pool(x)

        x = self.conv_dna.conv3(x)
        x = F.relu(x)
        a3 = x

        x = self.conv_dna.linear(x.view(-1, 53*180))
        x = F.relu(x)
        a4 = x
        x = F.sigmoid(self.linear(x))

        # calculate gradients mannually because onnx doesn't support backward
        grad = torch.ones(x.shape)

        grad *= x * (1. - x)

        grad = grad @ self.linear.weight
        grad *= (a4 > 0)

        grad = grad @ self.conv_dna.linear.weight
        grad = grad.view(1, 180, 53)
        grad *= (a3 > 0)

        # calculate gradcam heatmap
        gradcam_heatmap = F.relu((
            grad.sum(dim=2, keepdim=True) * grad
        ).sum(dim=1, keepdim=True))

        upsampled_heatmap = F.interpolate(
            gradcam_heatmap, size=(X.shape[-1],),
            mode='nearest')

        _grad = F.conv_transpose1d(grad, self.conv_dna.conv3.weight)
        grad = torch.zeros(a2.shape)
        grad = torch.scatter_add(grad, 2, indices_pool2, _grad)

        grad *= (a2 > 0)
        _grad = F.conv_transpose1d(grad, self.conv_dna.conv2.weight)

        grad = torch.zeros(a1.shape)
        grad = torch.scatter_add(grad, 2, indices_pool1, _grad)
        grad *= (a1 > 0)
        grad = F.conv_transpose1d(grad, self.conv_dna.conv1.weight)

        guided_grad_cam = grad * upsampled_heatmap

        guided_grad_cam = (guided_grad_cam - guided_grad_cam.min())
        guided_grad_cam /= guided_grad_cam.sum(dim=1).max()

        return x, guided_grad_cam


class DeepSeaModule(pl.LightningDataModule):

    def __init__(self, train_mat, val_mat, test_mat, batch_size=128, num_workers=16):
        super().__init__()
        self.train_mat = train_mat
        self.val_mat = val_mat
        self.test_mat = test_mat

        self.batch_size = batch_size
        self.num_workers = num_workers

    def _read_mat(self, mat_file, name):
        if name == 'train':
            f = h5py.File(mat_file, 'r')
            mat = {
                'trainxdata': np.einsum('sdb->bds', f['trainxdata'][:]),
                'traindata': np.einsum('ij->ji', f['traindata'][:])
            }
        else:
            mat = scipy.io.loadmat(mat_file)

        return mat[f'{name}xdata'], mat[f'{name}data']

    def _dataloader(self, mat_file, name):
        X, y = self._read_mat(mat_file, name)
        return DataLoader(
            TensorDataset(
                torch.tensor(X),
                torch.tensor(y),
            ),
            batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def train_dataloader(self):
        return self._dataloader(self.train_mat, 'train')

    def val_dataloader(self):
        return self._dataloader(self.val_mat, 'valid')

    def test_dataloader(self):
        return self._dataloader(self.test_mat, 'test')

