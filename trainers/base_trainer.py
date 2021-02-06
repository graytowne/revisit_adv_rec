import os
import time
from collections import OrderedDict

import numpy as np

from utils.utils import save_checkpoint, load_checkpoint


class BaseTrainer(object):
    def __init__(self):
        self.args = None

        self.n_users = None
        self.n_items = None

        self.net = None
        self.optimizer = None
        self.metrics = None
        self.golden_metric = "Recall@50"

    def __repr__(self):
        return self.args["model"]["model_name"] + "_" + str(self.net)

    @property
    def _initialized(self):
        return self.net is not None

    def _initialize(self):
        """Initialize model and optimizer."""
        # See actual implementation in each trainer.
        raise NotImplementedError

    def recommend(self, data, top_k, return_preds=False, allow_repeat=False):
        """Generate a top-k recommendation (ranked) list."""
        # See actual implementation in each trainer.
        raise NotImplementedError

    def train_epoch(self, data):
        """Train model for one epoch"""
        # See actual implementation in each trainer.
        raise NotImplementedError

    def train_epoch_wrapper(self, train_data, epoch_num):
        """Wrapper for train_epoch with some logs."""
        time_st = time.time()
        epoch_loss = self.train_epoch(train_data)
        print("Training [{:.1f} s], epoch: {}, loss: {:.4f}".format(
                time.time() - time_st, epoch_num, epoch_loss))

    def evaluate_epoch(self, train_data, test_data, epoch_num):
        """Evaluate model performance on test data."""
        t1 = time.time()

        n_rows = train_data.shape[0]
        n_evaluate_users = test_data.shape[0]

        total_metrics_len = sum(len(x) for x in self.metrics)
        total_val_metrics = np.zeros([n_rows, total_metrics_len], dtype=np.float32)

        recommendations = self.recommend(train_data, top_k=100)

        valid_rows = list()
        for i in range(train_data.shape[0]):
            # Ignore augmented users, evaluate only on real users.
            if i >= n_evaluate_users:
                continue
            targets = test_data[i].indices
            if targets.size <= 0:
                continue

            recs = recommendations[i].tolist()

            metric_results = list()
            for metric in self.metrics:
                result = metric(targets, recs)
                metric_results.append(result)
            total_val_metrics[i, :] = np.concatenate(metric_results)
            valid_rows.append(i)

        # Average evaluation results by user.
        total_val_metrics = total_val_metrics[valid_rows]
        avg_val_metrics = (total_val_metrics.mean(axis=0)).tolist()

        # Summary evaluation results into a dict.
        ind, result = 0, OrderedDict()
        for metric in self.metrics:
            values = avg_val_metrics[ind:ind + len(metric)]
            if len(values) <= 1:
                result[str(metric)] = values[0]
            else:
                for name, value in zip(str(metric).split(','), values):
                    result[name] = value
            ind += len(metric)

        print("Evaluation [{:.1f} s],  epoch: {}, {} ".format(
            time.time() - t1, epoch_num, str(result)))
        return result

    def validate(self, train_data, test_data, target_items):
        """Evaluate attack performance on target items."""
        t1 = time.time()

        n_rows = train_data.shape[0]
        n_evaluate_users = test_data.shape[0]

        # Init evaluation results.
        target_items_position = np.zeros([n_rows, len(target_items)], dtype=np.int64)

        recommendations = self.recommend(train_data, top_k=100)

        valid_rows = list()
        for i in range(train_data.shape[0]):
            # Ignore augmented users, evaluate only on real users.
            if i >= n_evaluate_users:
                continue
            targets = test_data[i].indices
            if targets.size <= 0:
                continue

            recs = recommendations[i].tolist()

            for j, item in enumerate(target_items):
                if item in recs:
                    target_items_position[i, j] = recs.index(item)
                else:
                    target_items_position[i, j] = train_data.shape[1]

            valid_rows.append(i)
        target_items_position = target_items_position[valid_rows]
        # Summary evaluation results into a dict.
        result = OrderedDict()
        result["TargetAvgRank"] = target_items_position.mean()
        # Note that here target_pos starts from 0.
        cutoff = 50
        result["TargetHR@%d" % cutoff] = (
            (target_items_position < cutoff).sum(1) >= 1).mean()

        # Log results.
        print("Attack Evaluation [{:.1f} s], {} ".format(
            time.time() - t1, str(result)))
        return result

    def fit(self, train_data, test_data):
        """Full model training loop."""
        if not self._initialized:
            self._initialize()

        if self.args.save_feq > self.args.epochs:
            raise ValueError("Model save frequency should be smaller than"
                             " total training epochs.")

        start_epoch = 1
        best_checkpoint_path = ""
        best_perf = 0.0
        for epoch_num in range(start_epoch, self.args.epochs + 1):
            # Train the model.
            self.train_epoch_wrapper(train_data, epoch_num)
            if epoch_num % self.args.save_feq == 0:
                result = self.evaluate_epoch(train_data, test_data, epoch_num)
                # Save model checkpoint if it has better performance.
                if result[self.golden_metric] > best_perf:
                    str_metric = "{}={:.4f}".format(self.golden_metric,
                                                    result[self.golden_metric])
                    print("Having better model checkpoint with"
                          " performance {}".format(str_metric))
                    checkpoint_path = os.path.join(
                        self.args.output_dir,
                        self.args.model['model_name'])
                    save_checkpoint(self.net, self.optimizer,
                                    checkpoint_path,
                                    epoch=epoch_num)

                    best_perf = result[self.golden_metric]
                    best_checkpoint_path = checkpoint_path

        # Load best model and evaluate on test data.
        print("Loading best model checkpoint.")
        self.restore(best_checkpoint_path)
        self.evaluate_epoch(train_data, test_data, -1)
        return

    def restore(self, path):
        """Restore model (and optimizer) state from checkpoint."""
        start_epoch, model_state, optimizer_state = load_checkpoint(path)
        self.net.load_state_dict(model_state)
        self.optimizer.load_state_dict(optimizer_state)
        return start_epoch
