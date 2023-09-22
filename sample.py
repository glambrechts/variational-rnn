import torch

from argparse import ArgumentParser

from data import mnist
from utils import show, auto_device
from model import VariationalRecurrentNeuralNetwork


WEIGHTS_DIRECTORY = 'weights'


def main(args):

    # Initialize model
    device = auto_device()
    vrnn = VariationalRecurrentNeuralNetwork(
        input_size=28,
        latent_size=args.latent_size,
        hidden_size=args.hidden_size,
        prior=not args.no_prior,
    ).to(device)

    # Load weights
    vrnn.load_state_dict(torch.load(f'{WEIGHTS_DIRECTORY}/{args.run_id}-best.pth'))

    # Sample
    while True:
        try:
            sample = vrnn.sample(8, 28, device=device).cpu()
            show(sample[:4], sample[4:], block=True)
        except KeyboardInterrupt:
            break


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('run_id')

    parser.add_argument('-Z', '--latent_size', type=int, default=32)
    parser.add_argument('-H', '--hidden_size', type=int, default=1024)

    parser.add_argument('-R', '--prior_weight', type=float, default=0.5)
    parser.add_argument('-Q', '--posterior_weight', type=float, default=0.1)
    parser.add_argument('-K', '--free_bits', type=float, default=1.0)
    parser.add_argument('-p', '--no_prior', action='store_true')

    parser.add_argument('-E', '--num_epochs', type=int, default=10)
    parser.add_argument('-B', '--batch_size', type=int, default=256)
    parser.add_argument('-a', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--clip_norm', type=float, default=100)

    args = parser.parse_args()

    main(args)
