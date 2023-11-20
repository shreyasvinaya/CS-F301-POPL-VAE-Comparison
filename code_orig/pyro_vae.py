import argparse
import itertools
import os
from abc import ABCMeta, abstractmethod
import time
from typing import Any
from code_orig.vae import VAE
import torch
import torch.nn as nn
from torch.nn import functional
from torchvision.utils import save_image
from code_external.mnist_cached import DATA_DIR, RESULTS_DIR

import pyro
from pyro.contrib.examples import util
from pyro.distributions import Bernoulli, Normal  # type: ignore
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam  # type: ignore

TRAIN = "train"
TEST = "test"
OUTPUT_DIR = RESULTS_DIR


class PyroVAEImpl(VAE):
    """Implemented by us using Pyro probabilistic programming framework."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = self.initialize_optimizer(lr=1e-3)

    def model(self, data):
        decoder = pyro.module("decoder", self.vae_decoder)
        z_mean, z_std = torch.zeros([data.size(0),
                                     20]), torch.ones([data.size(0), 20])
        with pyro.plate("data", data.size(0)):
            z = pyro.sample("latent", Normal(z_mean, z_std).to_event(1))
            img = decoder.forward(z)
            pyro.sample(
                "obs",
                Bernoulli(img, validate_args=False).to_event(1),
                obs=data.reshape(-1, 784),
            )

    def guide(self, data):
        encoder = pyro.module("encoder", self.vae_encoder)
        with pyro.plate("data", data.size(0)):
            z_mean, z_var = encoder.forward(data)
            pyro.sample("latent", Normal(z_mean, z_var.sqrt()).to_event(1))

    def compute_loss_and_gradient(self, x):
        if self.mode == TRAIN:
            loss = self.optimizer.step(x)
        else:
            loss = self.optimizer.evaluate_loss(x)
        loss /= self.args.batch_size * 784
        return loss

    def initialize_optimizer(self, lr):
        optimizer = Adam({"lr": lr})
        elbo = JitTrace_ELBO() if self.args.jit else Trace_ELBO()
        return SVI(self.model, self.guide, optimizer, loss=elbo)
