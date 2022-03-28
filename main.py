import argparse
from actor import DRL4Metro
from environment import Environment
from trainer import Trainer
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fair Network Expansion with Reinforcement Learning")

    parser.add_argument('--hidden_size', default=128, type=int)
    parser.add_argument('--static_size', default=2, type=int)
    parser.add_argument('--dynamic_size', default=1, type=int)

    args = parser.parse_args()

    # Prepare the environments.
    xian = Environment(Path('./environments/xian'))

    trainer_xian = Trainer(xian, )

    print("made it!")

    # Initialise trainer
    # trainer = Trainer()