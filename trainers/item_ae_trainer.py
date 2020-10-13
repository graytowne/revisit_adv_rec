import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from bunch import Bunch

from trainers.base_trainer import BaseTrainer
from trainers.losses import *
from utils.utils import sparse2tensor, minibatch


class ItemAE(nn.Module):
    def __init__(self, input_dim, model_args):
        super(ItemAE, self).__init__()
        self.q_dims = [input_dim] + model_args.hidden_dims
        self.p_dims = self.q_dims[::-1]

        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out
                                       in zip(self.q_dims[:-1], self.q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out
                                       in zip(self.p_dims[:-1], self.p_dims[1:])])

        self.recon_loss = model_args.recon_loss

    def encode(self, input):
        h = input
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            h = torch.tanh(h)
        return h

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = torch.tanh(h)
        return h

    def forward(self, input):
        z = self.encode(input)
        return self.decode(z)

    def loss(self, data, outputs):
        return self.recon_loss(data, outputs)


class ItemAETrainer(BaseTrainer):
    def __init__(self, n_users, n_items, args):
        super(ItemAETrainer, self).__init__()
        self.args = args

        self.device = torch.device("cuda" if self.args.use_cuda else "cpu")

        self.n_users = n_users
        self.n_items = n_items

        self.metrics = self.args.metrics

    def _initialize(self):
        model_args = Bunch(self.args.model)
        self.net = ItemAE(
            self.n_users, model_args=model_args).to(self.device)
        print(self)

        self.optimizer = optim.Adam(self.net.parameters(),
                                    lr=self.args.lr,
                                    weight_decay=self.args.l2)

    def train_epoch(self, data):
        # Transpose the data first for ItemVAE.
        data = data.transpose()

        n_rows = data.shape[0]
        n_cols = data.shape[1]
        idx_list = np.arange(n_rows)

        # Set model to training mode.
        model = self.net.to(self.device)
        model.train()
        np.random.shuffle(idx_list)

        epoch_loss = 0.0
        batch_size = (self.args.batch_size
                      if self.args.batch_size > 0 else len(idx_list))
        for batch_idx in minibatch(
                idx_list, batch_size=batch_size):
            batch_tensor = sparse2tensor(data[batch_idx]).to(self.device)

            # Compute loss
            outputs = model(batch_tensor)
            loss = model.loss(data=batch_tensor,
                              outputs=outputs).sum()
            epoch_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return epoch_loss

    def fit_adv(self, data_tensor, epoch_num, unroll_steps,
                n_fakes, target_items):
        import higher

        if not data_tensor.requires_grad:
            raise ValueError("To compute adversarial gradients, data_tensor "
                             "should have requires_grad=True.")

        self._initialize()

        data_tensor = data_tensor.to(self.device)
        target_tensor = torch.zeros_like(data_tensor)
        target_tensor[:, target_items] = 1.0
        data_tensor = data_tensor.t()

        n_rows = data_tensor.shape[0]
        n_cols = data_tensor.shape[1]
        idx_list = np.arange(n_rows)

        # Set model to training mode.
        model = self.net.to(self.device)
        optimizer = self.optimizer

        batch_size = (self.args.batch_size
                      if self.args.batch_size > 0 else len(idx_list))
        for i in range(1, epoch_num - unroll_steps + 1):
            t1 = time.time()
            np.random.shuffle(idx_list)
            model.train()
            epoch_loss = 0.0
            for batch_idx in minibatch(
                    idx_list, batch_size=batch_size):
                batch_tensor = data_tensor[batch_idx]
                # Compute loss
                outputs = model(batch_tensor)
                loss = model.loss(data=batch_tensor,
                                  outputs=outputs).sum()
                epoch_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print("Training [{:.1f} s], epoch: {}, loss: {:.4f}".format(
                time.time() - t1, i, epoch_loss))

        with higher.innerloop_ctx(model, optimizer) as (fmodel, diffopt):
            print("Switching to higher mode...")
            for i in range(epoch_num - unroll_steps + 1, epoch_num + 1):
                t1 = time.time()
                np.random.shuffle(idx_list)
                epoch_loss = 0.0
                fmodel.train()
                for batch_idx in minibatch(
                        idx_list, batch_size=batch_size):
                    batch_tensor = data_tensor[batch_idx]
                    # Compute loss
                    outputs = fmodel(batch_tensor)
                    loss = fmodel.loss(data=batch_tensor,
                                       outputs=outputs).sum()
                    epoch_loss += loss.item()
                    diffopt.step(loss)

                print("Training (higher mode) [{:.1f} s],"
                      " epoch: {}, loss: {:.4f}".format(time.time() - t1, i, epoch_loss))

            print("Finished surrogate model training,"
                  " {} copies of surrogate model params.".format(len(fmodel._fast_params)))

            fmodel.eval()
            all_preds = list()
            for batch_idx in minibatch(np.arange(n_rows),
                                       batch_size=batch_size):
                all_preds += [fmodel(data_tensor[batch_idx])]
            predictions = torch.cat(all_preds, dim=0).t()

            # Compute adversarial (outer) loss.
            adv_loss = mult_ce_loss(
                logits=predictions[:-n_fakes, ],
                data=target_tensor[:-n_fakes, ]).sum()
            adv_grads = torch.autograd.grad(adv_loss, data_tensor)[0]
            # Copy fmodel's parameters to default trainer.net().
            model.load_state_dict(fmodel.state_dict())

        return adv_loss.item(), adv_grads.t()[-n_fakes:, :]

    def recommend(self, data, top_k, return_preds=False, allow_repeat=False):
        # Set model to eval mode
        model = self.net.to(self.device)
        model.eval()

        # Transpose the data first for ItemVAE.
        data = data.transpose()

        n_rows = data.shape[0]
        n_cols = data.shape[1]

        idx_list = np.arange(n_rows)
        recommendations = np.empty([n_cols, top_k], dtype=np.int64)

        # Make predictions first, and then sort for top-k.
        all_preds = list()
        with torch.no_grad():
            for batch_idx in minibatch(
                    idx_list, batch_size=self.args.valid_batch_size):
                data_tensor = sparse2tensor(data[batch_idx]).to(self.device)
                preds = model(data_tensor)
                all_preds.append(preds)

        all_preds = torch.cat(all_preds, dim=0).t()
        data = data.transpose()
        idx_list = np.arange(n_cols)
        for batch_idx in minibatch(
                idx_list, batch_size=self.args.valid_batch_size):
            batch_data = data[batch_idx].toarray()
            preds = all_preds[batch_idx]
            if not allow_repeat:
                preds[batch_data.nonzero()] = -np.inf
            if top_k > 0:
                _, recs = preds.topk(k=top_k, dim=1)
                recommendations[batch_idx] = recs.cpu().numpy()

        if return_preds:
            return recommendations, all_preds.cpu()
        else:
            return recommendations
