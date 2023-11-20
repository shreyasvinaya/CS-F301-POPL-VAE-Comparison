import unittest
from code_orig.pyro_vae import PyroVAEImpl
from setup_vae import setup


class TestPyroVAEImpl(unittest.TestCase):

    def setUp(self):
        # Setup code, e.g., initializing a PyroVAEImpl instance
        args = {
            "num_epochs": 10,
            "batch_size": 128,
            "rng_seed": 0,
            "impl": 'pyro',
            "skip_eval": False,
            "jit": False
        }
        train_loader, test_loader = setup(args)
        self.pyro_vae = PyroVAEImpl(args, train_loader, test_loader)

    def test_initialization(self):
        # Test if the object is initialized correctly
        self.assertIsNotNone(self.pyro_vae.optimizer)


if __name__ == '__main__':
    unittest.main()
