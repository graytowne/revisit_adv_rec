import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from bunch import Bunch

from trainers.base_trainer import BaseTrainer
from utils.utils import sparse2tensor, minibatch


def _pairwise_jaccard(X):
    """Computes the Jaccard distance between the rows of `X`."""
    X = X.astype(bool).astype(np.uint16)

    intrsct = X.dot(X.T)
    row_sums = intrsct.diagonal()
    unions = (row_sums[:, None] + row_sums - intrsct).A
    dist = np.asarray(intrsct / unions)
    np.fill_diagonal(dist, 0.0)
    return dist


class ItemCF(nn.Module):
    def __init__(self, n_users, n_items, model_args):
        super(ItemCF, self).__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.knn = model_args.knn

        self.sims = nn.Parameter(
            torch.zeros([self.n_items, self.n_items]),
            requires_grad=False)

        self.top_nns = None
        self.top_sims = None

    def forward(self, item_id):
        if self.top_nns is None:
            self.top_sims, self.top_nns = self.sims.topk(k=self.knn, dim=1)
        return self.top_sims[item_id], self.top_nns[item_id]


class ItemCFTrainer(BaseTrainer):
    def __init__(self, n_users, n_items, args):
        super(ItemCFTrainer, self).__init__()
        self.args = args

        # ItemCF only works on CPU.
        # self.device = torch.device("cuda" if self.args.use_cuda else "cpu")

        self.n_users = n_users
        self.n_items = n_items

        self.metrics = self.args.metrics

    @property
    def _initialized(self):
        return self.net is not None

    def _initialize(self):
        model_args = Bunch(self.args.model)
        if not hasattr(model_args, "knn"):
            model_args.knn = 50
        self.net = ItemCF(
            n_users=self.n_users, n_items=self.n_items,
            model_args=model_args)
        print(self)

        # Set a fake optimizer for code consistency.
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr,
                                    weight_decay=self.args.l2)

    def train_epoch(self, data):
        self.net.sims.fill_(0.0)
        self.net.top_nns = None
        self.net.top_sims = None

        # Transpose the data first for ItemCF.
        train_data = data.transpose()

        self.net.sims.data = torch.FloatTensor(_pairwise_jaccard(train_data))

        return 0

    def recommend(self, data, top_k, return_preds=False, allow_repeat=False):
        model = self.net

        n_rows = data.shape[0]
        n_cols = data.shape[1]
        idx_list = np.arange(n_rows)

        nns_sims = torch.zeros([n_cols, n_cols])
        for item in range(n_cols):
            topk_sims, topk_nns = model(item_id=item)
            nns_sims[item].put_(topk_nns, topk_sims)

        recommendations = np.empty([n_rows, top_k], dtype=np.int64)
        all_preds = list()
        with torch.no_grad():
            for batch_idx in minibatch(
                    idx_list, batch_size=self.args.valid_batch_size):
                batch_tensor = sparse2tensor(data[batch_idx])

                preds = torch.mm(batch_tensor, nns_sims)
                if return_preds:
                    all_preds.append(preds)
                if not allow_repeat:
                    preds[data[batch_idx].nonzero()] = -np.inf
                if top_k > 0:
                    _, recs = preds.topk(k=top_k, dim=1)
                    recommendations[batch_idx] = recs.cpu().numpy()

        if return_preds:
            return recommendations, torch.cat(all_preds, dim=0).cpu()
        else:
            return recommendations
