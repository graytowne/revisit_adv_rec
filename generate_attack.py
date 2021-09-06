import argparse
import importlib

from bunch import Bunch

from data.data_loader import DataLoader
from utils.utils import set_seed, sample_target_items

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", type=str, default="generate_attack_args")
config = parser.parse_args()


def main(args):
    # Load data.
    print("Loading data from {}".format(args.data_path))
    data_loader = DataLoader(path=args.data_path)
    n_users, n_items = data_loader.n_users, data_loader.n_items
    print("n_users: {}, n_items: {}".format(n_users, n_items))
    train_data = data_loader.load_train_data()
    test_data = data_loader.load_test_data()

    # Train & evaluate adversarial users.
    attack_gen_args = Bunch(args.attack_gen_args)
    target_items = sample_target_items(
        train_data,
        n_samples=attack_gen_args.n_target_items,
        popularity=attack_gen_args.target_item_popularity,
        use_fix=attack_gen_args.use_fixed_target_item,
        output_dir=attack_gen_args.output_dir
    )
    attack_gen_args.target_items = target_items
    print(attack_gen_args)

    adv_trainer_class = attack_gen_args.trainer_class
    adv_trainer = adv_trainer_class(n_users=n_users,
                                    n_items=n_items,
                                    args=attack_gen_args)
    adv_trainer.fit(train_data, test_data)


if __name__ == "__main__":
    args = importlib.import_module(config.config_file)

    set_seed(args.seed, args.use_cuda)
    main(args)
