# Revisiting Adversarially Learned Injection Attacks Against Recommender Systems

A PyTorch implementation of paper:

*[Revisiting Adversarially Learned Injection Attacks Against Recommender Systems](https://arxiv.org/pdf/2008.04876.pdf), Jiaxi Tang, Hongyi Wen and Ke Wang , RecSys '20*

# Requirements
- Python 3.5+
- Please check [requirements.txt](https://github.com/graytowne/revisit_adv_rec/blob/master/requirements.txt) for required python packages.

# Running experiment
## Synthetic dataset 
[synthetic_exp.ipynb](https://github.com/graytowne/revisit_adv_rec/blob/master/synthetic_exp.ipynb) includes a self-contained implementation for:
1. Generating synthetic data
2. Experiments in the paper 

## Real-world dataset
Please refer to the following steps to run experiments on real-world dataset (i.e., Gowalla):
1. Install required packages.
2. Create a folder for experiment outputs (e.g., logs, model checkpoints, etc) 
    ```shell script
    cd revisit_adv_rec
    mkdir outputs
    ```
3. To generate fake data for attacking, change the configs in [generate_attack_args.py](https://github.com/graytowne/revisit_adv_rec/blob/master/generate_attack_args.py) (or leave as it is) then run:
     ```shell script
    python generate_attack.py
    ```
   You will find the fake data stored in the `outputs/` folder, such as `outputs/Sur-ItemAE_fake_data_best.npz`
4. To inject fake data and evaluate the recommender performance after attack, modify the configs in [evaluate_attack_args.py](https://github.com/graytowne/revisit_adv_rec/blob/master/evaluate_attack_args.py) (you need to point the `fake_data_path` to your own) then run:
     ```shell script
    python evaluate_attack.py
    ```

# Citation

If you use the code in your paper, please cite the paper:

```
@inproceedings{tang2020revisit,
  title={Revisiting Adversarially Learned Injection Attacks Against Recommender Systems},
  author={Tang, Jiaxi and Wen, Hongyi and Wang, Ke},
  booktitle={ACM Conference on Recommender Systems},
  year={2020}
}
```
