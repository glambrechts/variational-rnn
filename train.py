import os
import wandb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched

from tqdm import tqdm
from copy import deepcopy
from argparse import ArgumentParser

from data import mnist
from utils import show, auto_device
from model import VariationalRecurrentNeuralNetwork


WEIGHTS_DIRECTORY = 'weights'


def main(args):

    # Initialize weights and biases logging
    run = wandb.init(project='vrnn', config=args, save_code=True)
    config = run.config

    # Select device and load data
    device = auto_device()
    train_images, test_images = mnist(device=device)
    train_size = train_images.size(0)
    test_size = test_images.size(0)

    # Initialize model
    vrnn = VariationalRecurrentNeuralNetwork(
        input_size=train_images.size(-1),
        latent_size=config.latent_size,
        hidden_size=config.hidden_size,
        prior=not config.no_prior,
    ).to(device)

    # Initialize optimizer
    optimizer = optim.Adam(vrnn.parameters(), lr=config.learning_rate)
    scheduler = sched.ReduceLROnPlateau(optimizer, patience=5)

    # Save weights and statistics
    os.makedirs(WEIGHTS_DIRECTORY, exist_ok=True)
    best_weights = deepcopy(vrnn.state_dict())
    best_test = float('inf')

    # Free bits
    free_bits = 0.0 if args.free_bits is None else args.free_bits

    for epoch in range(config.num_epochs):

        print(f"Epoch {epoch:03d}")

        with torch.no_grad():

            # Test set
            mse, kl_prior, kl_posterior = vrnn(test_images)
            test_loss = mse
            mse += config.prior_weight * kl_prior.clamp(min=free_bits)
            mse += config.posterior_weight * kl_posterior.clamp(min=free_bits)

            scheduler.step(test_loss)

            # Keep best weights
            if test_loss < best_test:
                best_weights = deepcopy(vrnn.state_dict())
                best_test = test_loss

            torch.save(best_weights, f'{WEIGHTS_DIRECTORY}/{run.id}-best.pth')

            # Sample images
            samples = vrnn.sample(8, 28, device=device).cpu()
            samples = show(samples[:4], samples[4:])

            wandb.log({
                'epoch': epoch,
                'test_loss': test_loss.item(),
                'test_reconstruction': mse.item(),
                'test_regularization': kl_prior.item(),
                'sample': samples,
            })

            print(f"Test Loss: {test_loss:.4f}")

        permutation = torch.randperm(train_images.size(0))

        for i in tqdm(range(train_size // config.batch_size)):

            indices = permutation[i*config.batch_size : (i+1)*config.batch_size]
            inputs = train_images[indices, :, :]

            mse, kl_prior, kl_posterior = vrnn(inputs)
            loss = mse
            loss += config.prior_weight * kl_prior.clamp(min=free_bits)
            loss += config.posterior_weight * kl_posterior.clamp(min=free_bits)

            optimizer.zero_grad()
            loss.backward()
            if args.clip_norm is not None:
                nn.utils.clip_grad_norm_(vrnn.parameters(), args.clip_norm)
            optimizer.step()

            wandb.log({
                'epoch': epoch,
                'step': epoch * config.batch_size + i,
                'train_loss': loss.item(),
                'train_reconstruction': mse.item(),
                'train_regularization': kl_prior.item(),
            })

        torch.save(vrnn.state_dict(), f'{WEIGHTS_DIRECTORY}/{run.id}-final.pth')


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('-n', '--no_log', action='store_true')

    parser.add_argument('-Z', '--latent_size', type=int, default=32)
    parser.add_argument('-H', '--hidden_size', type=int, default=1024)

    parser.add_argument('-R', '--prior_weight', type=float, default=0.5)
    parser.add_argument('-Q', '--posterior_weight', type=float, default=0.1)
    parser.add_argument('-K', '--free_bits', type=float, default=1.0)
    parser.add_argument('-p', '--no_prior', action='store_true')

    parser.add_argument('-E', '--num_epochs', type=int, default=10)
    parser.add_argument('-B', '--batch_size', type=int, default=64)
    parser.add_argument('-a', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('-c', '--clip_norm', type=float, default=100)

    args = parser.parse_args()

    main(args)
