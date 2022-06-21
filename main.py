import argparse
import torch
import numpy as np

from constraints import ForwardConstraints
from environment import Environment
from trainer import Trainer
from pathlib import Path
from mlflow import log_metric, log_param, log_artifacts

def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

# torch.manual_seed(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fair Network Expansion with Reinforcement Learning")

    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--static_size', default=2, type=int)
    parser.add_argument('--dynamic_size', default=1, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--epoch_max', default=300, type=int)
    parser.add_argument('--train_size',default=128, type=int) # like a batch_size
    parser.add_argument('--line_unit_price', default=1.0, type=float)
    parser.add_argument('--station_price', default=5.0, type=float)
    parser.add_argument('--result_path', default=None, type=str)
    parser.add_argument('--actor_lr', default=10e-4, type=float)
    parser.add_argument('--critic_lr', default=10e-4, type=float)
    parser.add_argument('--station_num_lim', default=45, type=int)  # limit the number of stations in a line
    parser.add_argument('--budget', default=210, type=int)
    parser.add_argument('--max_grad_norm', default=2., type=float)
    parser.add_argument('--environment', default='diagonal_5x5', type=str)
    # reward types:
        # - weighted: a weighted sum of OD and equity reward -> --ses_weight * r_ses + (1-ses_weight) * r_od
        # - group: ODs are measured/regularized by group (see --groups_file), not a single OD.
        # - ai_economist: reward used by the ai_economist paper: total_utility * (1-gini(total_utility))
        # - rawls: returns the total satisfied OD of the lowest quintile group
    parser.add_argument('--reward', default='weighted', type=str)
    parser.add_argument('--ses_weight', default=0, type=float) # weight to assign to the socio-economic status (equity)reward, only works for --reward=weighted
    parser.add_argument('--var_lambda', default=0, type=float) # weight to assign to the variance of the satisfied OD among groups, only works for --reward=group

    parser.add_argument('--groups_file', default=None, type=str) # file that contains group membership of each grid square (e.g. when each square belongs to a certain income bin).

    parser.add_argument('--actor', choices=["pointer", "mlp", "cnn"], default="pointer", type=str)

    args = parser.parse_args()
    
    if args.seed:
        set_seed(args.seed)
    environment = Environment(Path(f"./environments/{args.environment}"), groups_file=args.groups_file)
    constraints = ForwardConstraints(environment.grid_x_size, environment.grid_y_size, environment.existing_lines_full, environment.grid_to_vector)
    trainer = Trainer(environment, constraints, args)

    # Log parameters on mlflow
    for arg, value in vars(args).items():
        log_param(arg, value)

    if not args.test: # Only train
        trainer.train(args)
    elif args.test and not args.result_path: # Train and test
        trainer.train(args)
        args.result_path = trainer.save_dir
        args.checkpoint_folder = trainer.checkpoint_dir
        trainer.evaluate(args)
    else: # Only test
        trainer.evaluate(args)

    print("made it!")
