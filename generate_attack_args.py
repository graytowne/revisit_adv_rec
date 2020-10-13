from functools import partial

from metrics.ranking_metrics import *
from trainers import *
from trainers.losses import *

data_path = "./data/gowalla"  # # Dataset path and loader
use_cuda = False  # If using GPU or CPU
seed = 1234  # Random seed
metrics = [PrecisionRecall(k=[50]), NormalizedDCG(k=[50])]

shared_params = {
    "use_cuda": use_cuda,
    "metrics": metrics,
    "seed": seed,
    "output_dir": "./outputs/",
}

""" Surrogate model hyper-parameters."""
sur_item_ae = {
    **shared_params,
    "epochs": 50,
    "lr": 1e-3,
    "l2": 1e-6,
    "save_feq": 50,
    "batch_size": 2048,
    "valid_batch_size": 512,
    "model": {
        "trainer_class": ItemAETrainer,
        "model_name": "Sur-ItemAE",
        "hidden_dims": [256, 128],
        "recon_loss": partial(mse_loss, weight=20)
    }
}
sur_wmf_sgd = {
    **shared_params,
    "epochs": 50,
    "lr": 1e-2,
    "l2": 1e-5,
    "save_feq": 50,
    "batch_size": 2048,
    "valid_batch_size": 512,
    "model": {
        "trainer_class": WMFTrainer,
        "model_name": "Sur-WeightedMF-sgd",
        "hidden_dims": [128],
        "weight_alpha": 20,
        "optim_method": "sgd"
    }
}
sur_wmf_als = {
    **shared_params,
    "epochs": 10,
    "lr": 1e-3,
    "l2": 5e-2,
    "save_feq": 10,
    "batch_size": 1,
    "valid_batch_size": 512,
    "model": {
        "trainer_class": WMFTrainer,
        "model_name": "Sur-WeightedMF-als",
        "hidden_dims": [128],
        "weight_alpha": 20,
        "optim_method": "als"
    }
}

""" Attack generation hyper-parameters."""
attack_gen_args = {
    **shared_params,
    "trainer_class": BlackBoxAdvTrainer,
    "attack_type": "adversarial",
    "n_target_items": 5,
    "target_item_popularity": "head",
    "use_fixed_target_item": True,

    # Args for adversarial training.
    "n_fakes": 0.01,
    "adv_epochs": 30,
    "unroll_steps": 1,

    "adv_lr": 1.0,
    "adv_momentum": 0.95,

    "proj_threshold": 0.05,
    "click_targets": False,

    # Args for surrogate model.
    "surrogate": sur_item_ae
}
