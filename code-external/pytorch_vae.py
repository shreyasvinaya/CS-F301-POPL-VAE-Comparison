import argparse
import itertools
import os
from abc import ABCMeta, abstractmethod
import time
from code_orig.VAE import VAE
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

class PyTorchVAEImpl(VAE):
    """
    Adapted from pytorch/examples.
    Source: https://github.com/pytorch/examples/tree/master/vae
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optimizer = self.initialize_optimizer(lr=1e-3)

    def compute_loss_and_gradient(self, x):
        self.optimizer.zero_grad()
        recon_x, z_mean, z_var = self.model_eval(x)
        binary_cross_entropy = functional.binary_cross_entropy(recon_x, x.reshape(-1, 784))
        # Uses analytical KL divergence expression for D_kl(q(z|x) || p(z))
        # Refer to Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # (https://arxiv.org/abs/1312.6114)
        kl_div = -0.5 * torch.sum(1 + z_var.log() - z_mean.pow(2) - z_var)
        kl_div /= self.args.batch_size * 784
        loss = binary_cross_entropy + kl_div
        if self.mode == TRAIN:
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def initialize_optimizer(self, lr=1e-3):
        model_params = itertools.chain(self.vae_encoder.parameters(),
                                       self.vae_decoder.parameters())
        return torch.optim.Adam(model_params, lr)