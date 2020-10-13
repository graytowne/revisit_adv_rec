import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from bunch import Bunch

from trainers.base_trainer import BaseTrainer
from trainers.losses import *
from utils.utils import sparse2tensor, minibatch


class UserVAE(nn.Module):
    def __init__(self, input_dim, model_args):
        super(UserVAE, self).__init__()
        self.q_dims = [input_dim] + model_args.hidden_dims
        self.p_dims = self.q_dims[::-1]
        # Double the latent code dimension for both mu and log_var in VAE.
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out
                                       in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out
                                       in zip(self.p_dims[:-1], self.p_dims[1:])])

        beta_init, self.beta_acc, self.beta_final = model_args.betas
        self.beta = beta_init

        self.recon_loss = model_args.recon_loss
        self.kld_loss = kld_loss

    def encode(self, input):
        h = input
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = torch.tanh(h)
            else:
                mu, log_var = torch.split(h, self.q_dims[-1], dim=1)
                return mu, log_var

    def reparameterize(self, mu, log_var):
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = torch.tanh(h)
        return h

    def forward(self, input, predict=False, **kwargs):
        # Annealing beta while training.
        if self.training:
            if self.beta_acc >= 0 and self.beta < self.beta_final:
                self.beta += self.beta_acc
            elif self.beta_acc < 0 and self.beta > self.beta_final:
                self.beta += self.beta_acc

        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        if predict:
            return self.decode(z)
        return self.decode(z), mu, log_var

    def loss(self, data, outputs):
        logits, mu, log_var = outputs
        return (self.recon_loss(data, logits) +
                self.beta * self.kld_loss(mu, log_var))


class CDAE(nn.Module):
    def __init__(self, input_dim, n_users, model_args):
        super(CDAE, self).__init__()
        self.q_dims = [input_dim] + model_args.hidden_dims
        self.p_dims = self.q_dims[::-1]

        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in
                                       zip(self.q_dims[:-1],
                                           self.q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
                                       d_in, d_out in
                                       zip(self.p_dims[:-1],
                                           self.p_dims[1:])])
        self.user_node = nn.Parameter(torch.randn([n_users, self.q_dims[-1]]),
                                      requires_grad=True)

        self.drop_input = torch.nn.Dropout(model_args.drop_rate)
        self.recon_loss = model_args.recon_loss

    def encode(self, input):
        h = input
        h = self.drop_input(h)
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            h = h * F.sigmoid(h)
        return h

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = F.selu(h)
        return h

    def forward(self, input, batch_user, **kwargs):
        z = self.encode(input) + self.user_node[batch_user]
        return self.decode(z)

    def loss(self, data, outputs):
        logits = outputs
        return self.recon_loss(data, logits)


class UserAETrainer(BaseTrainer):
    def __init__(self, n_users, n_items, args):
        super(UserAETrainer, self).__init__()
        self.args = args

        self.device = torch.device("cuda" if self.args.use_cuda else "cpu")

        self.n_users = n_users
        self.n_items = n_items

        self.metrics = self.args.metrics

    def _initialize(self):
        model_args = Bunch(self.args.model)
        if model_args.model_type == "UserVAE":
            if not hasattr(model_args, "recon_loss"):
                model_args.recon_loss = mult_ce_loss
            if not hasattr(model_args, "betas"):
                model_args.betas = [0.0, 1e-5, 1.0]
            self.net = UserVAE(
                self.n_items, model_args=model_args).to(self.device)
        elif model_args.model_type == "CDAE":
            if not hasattr(model_args, "recon_loss"):
                model_args.recon_loss = mult_ce_loss
            if not hasattr(model_args, "drop_rate"):
                model_args.drop_rate = 0.5
            self.net = CDAE(
                self.n_items, self.n_users,
                model_args=model_args).to(self.device)
        print(self)

        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.l2)

    def train_epoch(self, data):
        n_rows = data.shape[0]
        n_cols = data.shape[1]
        idx_list = np.arange(n_rows)

        # Set model to training mode.
        model = self.net.to(self.device)
        model.train()
        np.random.shuffle(idx_list)

        epoch_loss = 0.0
        counter = 0
        for batch_idx in minibatch(
                idx_list, batch_size=self.args.batch_size):
            batch_tensor = sparse2tensor(data[batch_idx]).to(self.device)

            # Compute loss
            outputs = model(batch_tensor,
                            batch_user=batch_idx)
            loss = model.loss(data=batch_tensor,
                              outputs=outputs).mean()
            epoch_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            counter += 1

        return epoch_loss / counter

    def recommend(self, data, top_k, return_preds=False, allow_repeat=False):
        # Set model to eval mode
        model = self.net.to(self.device)
        model.eval()

        n_rows = data.shape[0]
        idx_list = np.arange(n_rows)
        recommendations = np.empty([n_rows, top_k], dtype=np.int64)
        all_preds = list()
        with torch.no_grad():
            for batch_idx in minibatch(
                    idx_list, batch_size=self.args.valid_batch_size):
                batch_data = data[batch_idx]
                batch_tensor = sparse2tensor(batch_data).to(self.device)

                preds = model(batch_tensor,
                              batch_user=batch_idx,
                              predict=True)
                if return_preds:
                    all_preds.append(preds)
                if not allow_repeat:
                    preds[batch_data.nonzero()] = -np.inf
                if top_k > 0:
                    _, recs = preds.topk(k=top_k, dim=1)
                    recommendations[batch_idx] = recs.cpu().numpy()

        if return_preds:
            return recommendations, torch.cat(all_preds, dim=0).cpu()
        else:
            return recommendations
