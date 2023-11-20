import os
import pyro
from pyro.contrib.examples import util
from code_external.mnist_cached import DATA_DIR, RESULTS_DIR


def setup(args):
    pyro.set_rng_seed(args.rng_seed)
    train_loader = util.get_data_loader(
        dataset_name="MNIST",
        data_dir=DATA_DIR,
        batch_size=args.batch_size,
        is_training_set=True,
        shuffle=True,
    )
    test_loader = util.get_data_loader(
        dataset_name="MNIST",
        data_dir=DATA_DIR,
        batch_size=args.batch_size,
        is_training_set=False,
        shuffle=True,
    )
    global OUTPUT_DIR
    OUTPUT_DIR = os.path.join(RESULTS_DIR, args.impl)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    pyro.clear_param_store()
    return train_loader, test_loader
