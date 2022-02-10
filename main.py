import argparse
from environment import Environment
from trainer import Trainer
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fair Network Expansion with Reinforcement Learning")

    args = parser.parse_args()

    # Prepare the environments.
    xian = Environment(Path('./environments/xian'))

    print("made it!")

    # Initialise trainer
    # trainer = Trainer()