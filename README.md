# fair-network-expansion
 Fair network expansion with RL

## Command examples

### MLP
To train a MLP architecture with 6 layers for the actor, and 5 layers for the critic, on the diagonal environment run:

```
python main.py --seed 20 --environment diagonal_5x5 --groups_file groups.txt --arch mlp --actor_mlp_layers 6 --critic_mlp_layers 5 --epoch_max 60
```

To also test the model after training, add the test flag:

```
python main.py --seed 20 --environment diagonal_5x5 --groups_file groups.txt --arch mlp --actor_mlp_layers 6 --critic_mlp_layers 5 --epoch_max 60 --test
```

It is also possible to run the MLP architecture on the Xian environment:
```
python main.py --seed 20 --environment xian --groups_file price_groups_5.txt --arch mlp --actor_mlp_layers 6 --critic_mlp_layers 5 --epoch_max 300 --test
```

Adding attention to this model is done by changing `--arch mlp` to `--arch mlp-att`.

### CNN 
The CNN architecture is less modular, because it is harder to make the convolution sizes, number of layers, and grid size "click". To train and test a CNN model on the diagonal 5x5 grid run:
```
python main.py --seed 20 --environment diagonal_5x5 --groups_file groups.txt --arch cnn --epoch_max 60 --test
```

To run the CNN on the Xian environment, design a network that is able to process a 29x29 grid in `actor_modules.py` and in `critic.py`, and then run:
```
python main.py --seed 20 --environment xian --groups_file price_groups_5.txt --arch cnn --epoch_max 3500 --test
```
(There might be an old Xian CNN implementation somewhere in the commit history)

### RNN and Attention
The RNN achitecture can be run as follows:
```
python main.py --seed 20 --environment diagonal_5x5 --groups_file groups.txt --arch rnn --epoch_max 60 --test
```
Adding attention to this model is done by changing `--arch rnn` to `--arch rnn-att`.


## Constraint into reward function
To change the reward function to incorporate (some) constraints, use the argument `--constraint_free`. This will activate the reward scaling `--cf_reward_scaling` by default to be a linear function, since this is essential to the function of the model. This is based on the smooth line objective. Other options are the `--cf_efficient_station` (which should not be used) and the `--cf_station_density` arguments.