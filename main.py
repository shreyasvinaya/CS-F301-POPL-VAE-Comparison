import argparse
import pyro
from setup_vae import setup
import time
from code_external.pytorch_vae import PyTorchVAEImpl
from code_orig.pyro_vae import PyroVAEImpl


def main(args):
    train_loader, test_loader = setup(args)
    print("\n\n\n")
    if args.impl == "pyro":
        vae = PyroVAEImpl(args, train_loader, test_loader)
        print("Running Pyro VAE implementation")
    elif args.impl == "pytorch":
        vae = PyTorchVAEImpl(args, train_loader, test_loader)
        print("Running PyTorch VAE implementation")
    else:
        raise ValueError("Incorrect implementation specified: {}".format(
            args.impl))
    time1 = time.time()
    for i in range(args.num_epochs):
        vae.train(i)
        if not args.skip_eval:
            vae.test(i)
    time2 = time.time()
    print("\n\nTotal time taken: {:.2f}s".format(time2 - time1))


if __name__ == "__main__":
    assert pyro.__version__.startswith("1.8.6")
    parser = argparse.ArgumentParser(description="VAE using MNIST dataset")
    parser.add_argument("-n", "--num-epochs", nargs="?", default=10, type=int)
    parser.add_argument("--batch_size", nargs="?", default=128, type=int)
    parser.add_argument("--rng_seed", nargs="?", default=0, type=int)
    parser.add_argument("--impl", nargs="?", default="pyro", type=str)
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--jit", action="store_true")
    parser.set_defaults(skip_eval=False)
    args = parser.parse_args()
    print(args)
    main(args)
