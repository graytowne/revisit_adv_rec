import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from bunch import Bunch

from trainers.base_trainer import BaseTrainer
from trainers.losses import *
from utils.utils import sparse2tensor, tensor2sparse, minibatch


def _array2sparsediag(x):
    values = x
    indices = np.vstack([np.arange(x.size), np.arange(x.size)])

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = [x.size, x.size]

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


class WeightedMF(nn.Module):
    def __init__(self, n_users, n_items, model_args):
        super(WeightedMF, self).__init__()

        hidden_dims = model_args.hidden_dims
        if len(hidden_dims) > 1:
            raise ValueError("WMF can only have one latent dimension.")

        self.n_users = n_users
        self.n_items = n_items
        self.dim = hidden_dims[0]

        self.Q = nn.Parameter(
            torch.zeros([self.n_items, self.dim]).normal_(mean=0, std=0.1))
        self.P = nn.Parameter(
            torch.zeros([self.n_users, self.dim]).normal_(mean=0, std=0.1))
        self.params = nn.ParameterList([self.Q, self.P])

    def forward(self, user_id=None, item_id=None):
        if user_id is None and item_id is None:
            return torch.mm(self.P, self.Q.t())
        if user_id is not None:
            return torch.mm(self.P[[user_id]], self.Q.t())
        if item_id is not None:
            return torch.mm(self.P, self.Q[[item_id]].t())


class WMFTrainer(BaseTrainer):
    def __init__(self, n_users, n_items, args):
        super(WMFTrainer, self).__init__()
        self.args = args

        self.device = torch.device("cuda" if self.args.use_cuda else "cpu")

        self.n_users = n_users
        self.n_items = n_items

        self.metrics = self.args.metrics

    def _initialize(self):
        model_args = Bunch(self.args.model)
        if not hasattr(model_args, "optim_method"):
            model_args.optim_method = "sgd"
        self.net = WeightedMF(
            n_users=self.n_users, n_items=self.n_items,
            model_args=model_args).to(self.device)
        print(self)

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr,
                                    weight_decay=self.args.l2)

        self.optim_method = model_args.optim_method
        self.weight_alpha = self.args.model["weight_alpha"]
        self.dim = self.net.dim

    def train_epoch(self, *args, **kwargs):
        if self.optim_method not in ("sgd", "als"):
            raise ValueError("Unknown optim_method {} for WMF.".format(
                self.optim_method))

        if self.optim_method == "sgd":
            return self.train_sgd(*args, **kwargs)
        if self.optim_method == "als":
            return self.train_als(*args, **kwargs)

    def train_als(self, data):
        model = self.net  # A warning will raise if use .to() method here.
        P = model.P.detach()
        Q = model.Q.detach()

        weight_alpha = self.weight_alpha - 1
        # Using PyTorch for ALS optimization
        # Update P
        lamda_eye = torch.eye(self.dim).to(self.device) * self.args.l2
        # residual = Q^tQ + lambda*I
        residual = torch.mm(Q.t(), Q) + lamda_eye
        for user, batch_data in enumerate(data):
            # x_u: N x 1
            x_u = sparse2tensor(batch_data).to(self.device).t()
            # Cu = diagMat(alpha * rating + 1)
            cu = batch_data.toarray().squeeze() * weight_alpha + 1
            Cu = _array2sparsediag(cu).to(self.device)
            Cu_minusI = _array2sparsediag(cu - 1).to(self.device)
            # Q^tCuQ + lambda*I = Q^tQ + lambda*I + Q^t(Cu-I)Q
            # left hand side
            lhs = torch.mm(Q.t(), Cu_minusI.mm(Q)) + residual
            # right hand side
            rhs = torch.mm(Q.t(), Cu.mm(x_u))

            new_p_u = torch.mm(lhs.inverse(), rhs)
            model.P.data[user] = new_p_u.t()

        # Update Q
        data = data.transpose()
        # residual = P^tP + lambda*I
        residual = torch.mm(P.t(), P) + lamda_eye
        for item, batch_data in enumerate(data):
            # x_v: M x 1
            x_v = sparse2tensor(batch_data).to(self.device).t()
            # Cv = diagMat(alpha * rating + 1)
            cv = batch_data.toarray().squeeze() * weight_alpha + 1
            Cv = _array2sparsediag(cv).to(self.device)
            Cv_minusI = _array2sparsediag(cv - 1).to(self.device)
            # left hand side
            lhs = torch.mm(P.t(), Cv_minusI.mm(P)) + residual
            # right hand side
            rhs = torch.mm(P.t(), Cv.mm(x_v))

            new_q_v = torch.mm(lhs.inverse(), rhs)
            model.Q.data[item] = new_q_v.t()

        return 0

    def train_sgd(self, data):
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
            outputs = model(user_id=batch_idx)
            loss = mse_loss(data=batch_tensor,
                            logits=outputs,
                            weight=self.weight_alpha).sum()
            epoch_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return epoch_loss

    def fit_adv(self, *args, **kwargs):
        self._initialize()

        if self.optim_method not in ("sgd", "als"):
            raise ValueError("Unknown optim_method {} for WMF.".format(
                self.optim_method))

        if self.optim_method == "sgd":
            return self.fit_adv_sgd(*args, **kwargs)
        if self.optim_method == "als":
            return self.fit_adv_als(*args, **kwargs)

    def fit_adv_sgd(self, data_tensor, epoch_num, unroll_steps,
                    n_fakes, target_items):
        import higher

        if not data_tensor.requires_grad:
            raise ValueError("To compute adversarial gradients, data_tensor "
                             "should have requires_grad=True.")

        data_tensor = data_tensor.to(self.device)
        target_tensor = torch.zeros_like(data_tensor)
        target_tensor[:, target_items] = 1.0
        n_rows = data_tensor.shape[0]
        n_cols = data_tensor.shape[1]
        idx_list = np.arange(n_rows)

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
                # Compute loss
                loss = mse_loss(data=data_tensor[batch_idx],
                                logits=model(user_id=batch_idx),
                                weight=self.weight_alpha).sum()
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
                fmodel.train()
                epoch_loss = 0.0
                for batch_idx in minibatch(
                        idx_list, batch_size=batch_size):
                    # Compute loss
                    loss = mse_loss(data=data_tensor[batch_idx],
                                    logits=fmodel(user_id=batch_idx),
                                    weight=self.weight_alpha).sum()
                    epoch_loss += loss.item()
                    diffopt.step(loss)

                print("Training (higher mode) [{:.1f} s],"
                      " epoch: {}, loss: {:.4f}".format(time.time() - t1, i, epoch_loss))

            print("Finished surrogate model training,"
                  " {} copies of surrogate model params.".format(len(fmodel._fast_params)))

            fmodel.eval()
            predictions = fmodel()
            # Compute adversarial (outer) loss.
            adv_loss = mult_ce_loss(
                logits=predictions[:-n_fakes, ],
                data=target_tensor[:-n_fakes, ]).sum()
            adv_grads = torch.autograd.grad(adv_loss, data_tensor)[0]
            # Copy fmodel's parameters to default trainer.net().
            model.load_state_dict(fmodel.state_dict())

        return adv_loss.item(), adv_grads[-n_fakes:, ]

    def fit_adv_als(self, data_tensor, epoch_num, unroll_steps,
                    n_fakes, target_items):
        if not data_tensor.requires_grad:
            raise ValueError("To compute adversarial gradients, data_tensor "
                             "should have requires_grad=True.")

        train_data = tensor2sparse(data_tensor)
        for i in range(1, epoch_num + 1):
            t1 = time.time()
            self.train_als(data=train_data)
            print("Training [{:.1f} s], epoch: {}".format(time.time() - t1, i))

        data_tensor = data_tensor.to(self.device)
        target_tensor = torch.zeros_like(data_tensor)
        target_tensor[:, target_items] = 1.0

        model = self.net.to(self.device)
        model.eval()

        # Compute adversarial (outer) loss.
        predictions = model()
        adv_loss = mult_ce_loss(
            logits=predictions[:-n_fakes, ],
            data=target_tensor[:-n_fakes, ]).sum()
        dloss_dpreds = torch.autograd.grad(adv_loss, predictions)[0]

        # Transpose both data_tensor and dloss_dpreds for easy indexing later.
        data_tensor = data_tensor.t()
        dloss_dpreds = dloss_dpreds.t()

        P = model.P.detach()

        weight_alpha = self.weight_alpha - 1
        lamda_eye = torch.eye(self.dim).to(self.device) * self.args.l2
        # residual = P^tP + lambda*I
        residual = torch.mm(P.t(), P) + lamda_eye

        adv_grads = list()
        for i, x_v in enumerate(data_tensor):
            # Cv = diagMat(alpha * rating + 1)
            cv = x_v.detach().cpu().numpy() * weight_alpha + 1
            Cv = _array2sparsediag(cv).to(self.device)
            Cv_minusI = _array2sparsediag(cv - 1).to(self.device)
            # left hand side
            lhs = torch.mm(P.t(), Cv_minusI.mm(P)) + residual
            # right hand side
            rhs = torch.mm(P.t(), Cv.mm(x_v.view(-1, 1)))

            new_q_v = torch.mm(lhs.inverse(), rhs)
            r_v = torch.mm(P, new_q_v)

            adv_grad = torch.autograd.grad(r_v, x_v, dloss_dpreds[i].view_as(r_v))[0]
            adv_grads.append(adv_grad.view(-1, 1))
        adv_grads = torch.cat(adv_grads, dim=1)

        return adv_loss.item(), adv_grads[-n_fakes:, :]

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
                batch_data = data[batch_idx].toarray()

                preds = model(user_id=batch_idx)
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
