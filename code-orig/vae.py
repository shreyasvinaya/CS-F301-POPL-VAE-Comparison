import argparse
import itertools
import os
from abc import ABCMeta, abstractmethod
import time
from typing import Any

import torch
import torch.nn as nn
from torch.nn import functional
from torchvision.utils import save_image
from mnist_cached import DATA_DIR, RESULTS_DIR

import pyro
from pyro.contrib.examples import util
from pyro.distributions import Bernoulli, Normal
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam

TRAIN = "train"
TEST = "test"
OUTPUT_DIR = RESULTS_DIR


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.relu = nn.ReLU()

    def forward(self, x) -> tuple[Any, torch.Tensor]:
        x = x.reshape(-1, 784)
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), torch.exp(self.fc22(h1))


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)
        self.relu = nn.ReLU()

    def forward(self, z):
        h3 = self.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))


class VAE(object, metaclass=ABCMeta):

    def __init__(self, args, train_loader, test_loader):
        self.args = args
        self.vae_encoder = Encoder()
        self.vae_decoder = Decoder()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.mode = TRAIN

    def set_train(self, is_train=True):
        if is_train:
            self.mode = TRAIN
            self.vae_encoder.train()
            self.vae_decoder.train()
        else:
            self.mode = TEST
            self.vae_encoder.eval()
            self.vae_decoder.eval()

    @abstractmethod
    def compute_loss_and_gradient(self, x):
        return

    def model_eval(self, x):
        z_mean, z_var = self.vae_encoder(x)
        if self.mode == TRAIN:
            z = Normal(z_mean, z_var.sqrt()).rsample()
        else:
            z = z_mean
        return self.vae_decoder(z), z_mean, z_var

    def train(self, epoch):
        self.set_train(is_train=True)
        train_loss = 0
        for batch_idx, (x, _) in enumerate(self.train_loader):
            loss = self.compute_loss_and_gradient(x)
            train_loss += loss
        print("====> Epoch: {} \nTraining loss: {:.4f}".format(
            epoch, train_loss / len(self.train_loader.dataset)))

    def test(self, epoch):
        self.set_train(is_train=False)
        test_loss = 0
        for i, (x, _) in enumerate(self.test_loader):
            with torch.no_grad():
                recon_x = self.model_eval(x)[0]
                test_loss += self.compute_loss_and_gradient(x)
            if i == 0:
                n = min(x.size(0), 8)
                comparison = torch.cat([
                    x[:n],
                    recon_x.reshape(self.args.batch_size, 1, 28, 28)[:n]
                ])
                save_image(
                    comparison.detach().cpu(),
                    os.path.join(OUTPUT_DIR,
                                 "reconstruction_" + str(epoch) + ".png"),
                    nrow=n,
                )

        test_loss /= len(self.test_loader.dataset)
        print("Test set loss: {:.4f}".format(test_loss))