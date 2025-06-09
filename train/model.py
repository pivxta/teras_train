import torch
import numpy
import pytorch_lightning as pl
from torch import nn
from feature import FEATURE_COUNT 

FT_OUT = 256

ACTIVATION_RANGE = 127
WEIGHT_SCALING = 64
OUTPUT_WEIGHT_SCALING = 32
OUTPUT_SCALING = 300
WEIGHT_MAX_HIDDEN = ACTIVATION_RANGE / WEIGHT_SCALING
WEIGHT_MIN_HIDDEN = -WEIGHT_MAX_HIDDEN
WEIGHT_MAX_OUTPUT = (ACTIVATION_RANGE * ACTIVATION_RANGE) / (OUTPUT_WEIGHT_SCALING * OUTPUT_SCALING)
WEIGHT_MIN_OUTPUT = -WEIGHT_MAX_OUTPUT

def quantize_ft_weights(tensor: torch.Tensor):
    return numpy.round(tensor.detach().numpy().transpose() * ACTIVATION_RANGE).astype('<i2').flatten()

def quantize_ft_biases(tensor: torch.Tensor):
    return numpy.round(tensor.detach().numpy() * ACTIVATION_RANGE).astype('<i2').flatten()

def quantize_weights(tensor: torch.Tensor):
    return numpy.round(
        tensor
            .detach()
            .clamp(WEIGHT_MIN_HIDDEN, WEIGHT_MAX_HIDDEN)
            .numpy() * WEIGHT_SCALING
        ).astype('<i1').flatten()

def quantize_biases(tensor: torch.Tensor):
    return numpy.round(
        tensor.detach().numpy() * WEIGHT_SCALING * ACTIVATION_RANGE
        ).astype('<i4').flatten()

def quantize_output_weights(tensor: torch.Tensor):
    return numpy.round(
        tensor
            .detach()
            .clamp(WEIGHT_MIN_OUTPUT, WEIGHT_MAX_OUTPUT)
            .numpy() * OUTPUT_WEIGHT_SCALING * OUTPUT_SCALING / ACTIVATION_RANGE
        ).astype('<i1').flatten()

def quantize_output_biases(tensor: torch.Tensor):
    return numpy.round(
        tensor.detach().numpy() * OUTPUT_WEIGHT_SCALING * OUTPUT_SCALING
        ).astype('<i4').flatten()

def cross_entropy_loss(target, prediction):
    epsilon = 1e-9
    bce = target * torch.log(target + epsilon) + (1 - target) * torch.log(1 - target + epsilon) \
        - target * torch.log(prediction + epsilon) - (1 - target) * torch.log(1 - prediction + epsilon)
    return bce.mean()


class NNUE(pl.LightningModule):
    def __init__(self, lr, eval_weight):
        super().__init__()
        self.lr = lr
        self.eval_weight = eval_weight

        self.ft = nn.Linear(FEATURE_COUNT, FT_OUT) 
        self.hidden1 = nn.Linear(FT_OUT * 2, 16)
        self.hidden2 = nn.Linear(16, 32)
        self.out = nn.Linear(32, 1)
        self.weight_clipping = [
            {'params': [self.hidden1.weight], 'min_weight': WEIGHT_MIN_HIDDEN, 'max_weight': WEIGHT_MAX_HIDDEN},
            {'params': [self.hidden2.weight], 'min_weight': WEIGHT_MIN_HIDDEN, 'max_weight': WEIGHT_MAX_HIDDEN},
            {'params': [self.out.weight], 'min_weight': WEIGHT_MIN_OUTPUT, 'max_weight': WEIGHT_MAX_OUTPUT},
        ]

    def forward(self, batch):
        stm_out: torch.Tensor = self.ft(batch.stm_features)
        non_stm_out: torch.Tensor = self.ft(batch.non_stm_features)
        hidden = torch.clamp(torch.cat((stm_out, non_stm_out), dim=1), 0.0, 1.0)
        hidden = torch.clamp(self.hidden1(hidden), 0.0, 1.0)
        hidden = torch.clamp(self.hidden2(hidden), 0.0, 1.0)
        return self.out(hidden)

    def _clip_weights(self):
        for group in self.weight_clipping:
            min_weight = group['min_weight']
            max_weight = group['max_weight']
            for params in group['params']:
                data = params.data
                data = data.clamp(min_weight, max_weight)
                params.data = data

    def _step(self, batch, batch_idx):
        self._clip_weights()
        target_scaling = 400
        prediction = torch.sigmoid(self(batch))
        target_eval = torch.sigmoid(batch.evals / target_scaling)
        target_outcome = batch.outcomes

        loss_eval = cross_entropy_loss(target_eval, prediction)
        loss_outcome = cross_entropy_loss(target_outcome, prediction)
        loss = self.eval_weight * loss_eval + (1.0 - self.eval_weight) * loss_outcome

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        target_scaling = 400
        prediction = torch.sigmoid(self(batch))
        target_eval = torch.sigmoid(batch.evals / target_scaling)
        target_outcome = batch.outcomes

        loss_eval = cross_entropy_loss(target_eval, prediction)
        loss_outcome = cross_entropy_loss(target_outcome, prediction)

        self.log('val_loss_eval', loss_eval, on_epoch=True, prog_bar=True)
        self.log('val_loss_wdl', loss_outcome, on_epoch=True, prog_bar=True)
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def write_to_file(self, filename: str):
        with open(filename, "wb") as file:
            file.write(bytes(memoryview(quantize_ft_weights(self.ft.weight))))
            file.write(bytes(memoryview(quantize_ft_biases(self.ft.bias))))
            file.write(bytes(memoryview(quantize_weights(self.hidden1.weight))))
            file.write(bytes(memoryview(quantize_biases(self.hidden1.bias))))
            file.write(bytes(memoryview(quantize_weights(self.hidden2.weight))))
            file.write(bytes(memoryview(quantize_biases(self.hidden2.bias))))
            file.write(bytes(memoryview(quantize_output_weights(self.out.weight))))
            file.write(bytes(memoryview(quantize_output_biases(self.out.bias))))

