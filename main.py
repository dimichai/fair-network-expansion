import argparse

import torch
from actor import DRL4Metro
from constraints import ForwardConstraints
from environment import Environment
from trainer import Trainer
from pathlib import Path

# torch.manual_seed(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fair Network Expansion with Reinforcement Learning")

    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--static_size', default=2, type=int)
    parser.add_argument('--dynamic_size', default=1, type=int)
    parser.add_argument('--num_layers', default=1, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--epoch_max', default=3500, type=int)
    parser.add_argument('--train_size',default=128, type=int) # like a batch_size
    parser.add_argument('--line_unit_price', default=1.0, type=float)
    parser.add_argument('--station_price', default=5.0, type=float)
    parser.add_argument('--result_path', default=Path('./result'), type=str)
    parser.add_argument('--actor_lr', default=10e-4, type=float)
    parser.add_argument('--critic_lr', default=10e-4, type=float)
    parser.add_argument('--station_num_lim', default=45, type=int)  # limit the number of stations in a line
    parser.add_argument('--budget', default=210, type=int)
    parser.add_argument('--max_grad_norm', default=2., type=float)

    args = parser.parse_args()

    # Prepare the environments.
    xian = Environment(Path('./environments/xian'))
    # diag = Environment(Path('./environments/diagonal_5x5'))

    constraints = ForwardConstraints(xian.grid_x_size, xian.grid_y_size, xian.existing_lines_full, xian.grid_to_vector)

    trainer_xian = Trainer(xian, constraints, args)

    # TODO: maybe make it so that if there is a checkpoint, training log continues from that epoch and not from the start
    if not args.test:
        trainer_xian.train(args)
        

    print("made it!")
