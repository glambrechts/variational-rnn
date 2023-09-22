import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.distributions as dist

from torch.distributions.kl import kl_divergence


class Split(nn.Module):

    def __init__(self, half_size, m1=None, m2=None):

        super().__init__()

        self.half_size = half_size
        self.m1 = nn.Identity() if m1 is None else m1
        self.m2 = nn.Identity() if m2 is None else m2

    def forward(self, x):

        x1 = x[..., :self.half_size]
        x2 = x[..., self.half_size:]
        return self.m1(x1), self.m2(x2)


class Reshape(nn.Module):

    def __init__(self, *shape):

        super().__init__()

        self.shape = shape

    def forward(self, x):

        return x.view(*self.shape)


class VariationalRecurrentNeuralNetwork(nn.Module):

    def __init__(
            self,
            input_size=28,
            latent_size=32,
            num_classes=32,
            gumbel_temperature=1.0,
            hidden_size=256,
            prior=False
        ):

        super().__init__()

        self.input_size = input_size
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.gumbel_temperature = gumbel_temperature
        self.hidden_size = hidden_size

        self.epsilon = torch.finfo(torch.float).eps
        self.two_pi = torch.tensor(2.0 * torch.pi)

        # Features
        self.x_feature = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ELU(),
            nn.Linear(hidden_size, hidden_size), nn.ELU(),
        )

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size + hidden_size, hidden_size), nn.ELU(),
            nn.Linear(hidden_size, hidden_size), nn.ELU(),
            nn.Linear(hidden_size, latent_size * num_classes),
            Reshape(-1, latent_size, num_classes),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_size * num_classes + hidden_size, hidden_size), nn.ELU(),
            nn.Linear(hidden_size, hidden_size), nn.ELU(),
            nn.Linear(hidden_size, input_size),
        )

        # Prior
        if prior:
            self.prior = nn.Sequential(
                nn.Linear(hidden_size, hidden_size), nn.ELU(),
                nn.Linear(hidden_size, latent_size * num_classes),
                Reshape(-1, latent_size, num_classes)
            )
        else:
            self.prior = None

        # Recurrence
        self.rnn = nn.GRU(
            latent_size * num_classes, hidden_size,
            num_layers=1, batch_first=True,
        )

    def forward(self, x):

        batch_size = x.size(0)
        seq_length = x.size(1)

        h = torch.zeros(batch_size, self.hidden_size, device=x.device)

        mse = 0.0
        kl_prior = 0.0
        kl_posterior = 0.0

        for t in range(seq_length):

            # Prior
            if self.prior is None:
                logits_p = torch.ones(batch_size, self.latent_size, self.num_classes, device=x.device)
            else:
                logits_p = self.prior(h)
            prior_dist = dist.Categorical(logits=logits_p)
            detached_prior_dist = dist.Categorical(logits=logits_p.detach())

            # Current input
            xx = self.x_feature(x[:, t, :])

            # Encoder
            logits_e = self.encoder(torch.cat((xx, h), dim=-1))
            latent_dist = dist.Categorical(logits=logits_e)
            detached_latent_dist = dist.Categorical(logits=logits_e.detach())

            # Sample latent (with Gumbel reparametrization trick)
            z = func.one_hot(latent_dist.sample(), num_classes=self.num_classes).float()
            probs = func.softmax(latent_dist.logits / self.gumbel_temperature, dim=-1)
            z = z + probs - probs.detach()
            z = z.view(-1, self.latent_size * self.num_classes)

            # Decoder
            mu_d = self.decoder(torch.cat((z, h), dim=-1))

            # Recurrence
            _, h = self.rnn(z.unsqueeze(1), h.unsqueeze(0))
            h = h.squeeze(0)

            # Losses
            mse += (
                (x[:, t, :] - mu_d) ** 2
            ).sum(dim=-1).mean(dim=0)
            kl_prior += kl_divergence(
                detached_latent_dist,
                prior_dist,
            ).sum(dim=-1).mean(dim=0)
            kl_posterior += kl_divergence(
                latent_dist,
                detached_prior_dist,
            ).sum(dim=-1).mean(dim=0)

        return mse, kl_prior, kl_posterior

    def sample(self, batch_size, seq_length, device='cpu'):

        h = torch.zeros(batch_size, self.hidden_size, device=device)

        outputs = []

        with torch.no_grad():

            for t in range(seq_length):

                # Prior
                if self.prior is None:
                    logits_p = torch.ones(batch_size, self.latent_size, self.num_classes, device=device)
                else:
                    logits_p = self.prior(h)
                prior_dist = dist.Categorical(logits=logits_p)

                # Sample latent
                z = func.one_hot(prior_dist.sample(), num_classes=self.num_classes).float()
                z = z.view(-1, self.latent_size * self.num_classes)

                # Decoder
                mu_d = self.decoder(torch.cat((z, h), dim=-1))
                outputs.append(mu_d)

                # Recurrence
                _, h = self.rnn(z.unsqueeze(1), h.unsqueeze(0))
                h = h.squeeze(0)

        return torch.stack(outputs, dim=1)
