import argparse

import torch
from actor import DRL4Metro
from environment import Environment
from trainer import Trainer
from pathlib import Path
from environments.xian.constraints import allowed_next_vector_indices

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

    args = parser.parse_args()

    # Prepare the environments.
    xian = Environment(Path('./environments/xian'))

    trainer_xian = Trainer(xian, args)

    # dir_v, allowed_v = allowed_next_vector_indices(242, torch.Tensor([[8, 10]]).long(), torch.Tensor([[8, 11]]).long(), torch.Tensor([[0, 0, 0, 0, 1, 0, 1, 0]]).long())
    # print(dir_v, allowed_v)
    # # correct is tensor([[0, 0, 0, 0, 1, 0, 1, 0]]), tensor([240, 241, 269, 270, 271, 298, 299, 300])

    # dir_v, allowed_v = allowed_next_vector_indices(310, torch.Tensor([[10, 20]]).long(), torch.Tensor([[8, 20]]).long(), torch.Tensor([[0, 0, 0, 0, 0, 0, 0, 0]]).long())
    # print(dir_v, allowed_v)
    # # correct is tensor([[0, 0, 0, 0, 1, 0, 0, 0]]), tensor([308, 309, 311, 312, 337, 338, 339, 340, 341, 366, 367, 368, 369, 370])

    print("made it!")

    # Initialise trainer
    # trainer = Trainer()