import unittest
from code_external.pytorch_vae import PyTorchVAEImpl
from code_orig.pyro_vae import PyroVAEImpl
from setup_vae import setup
import argparse


class TestPyroVAEImpl(unittest.TestCase):

    def setUp(self):
        # Setup code
        parser = argparse.ArgumentParser(description="VAE using MNIST dataset")
        parser.add_argument("-n",
                            "--num-epochs",
                            nargs="?",
                            default=10,
                            type=int)
        parser.add_argument("--batch_size", nargs="?", default=128, type=int)
        parser.add_argument("--rng_seed", nargs="?", default=0, type=int)
        parser.add_argument("--impl", nargs="?", default="pyro", type=str)
        parser.add_argument("--skip_eval", action="store_true")
        parser.add_argument("--jit", action="store_true")
        parser.set_defaults(skip_eval=False)
        args = parser.parse_args()
        train_loader, test_loader = setup(args)
        self.pyro_vae = PyroVAEImpl(args, train_loader, test_loader)

    def test_initialization(self):
        # Test if the object is initialized correctly
        self.assertIsNotNone(self.pyro_vae.optimizer)


if __name__ == '__main__':
    unittest.main()
