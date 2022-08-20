# Rewarding rationality to improve reinforcment learning models on the metro network design problem
This is the repository for the accompanying code of the master project "Rewarding rationality to improve reinforcement learning models on the metro network design problem. This readme includes the steps to reproduce the results, as well as a general overview of the functionality of the code.

What follows is a quick guide reproduce the results. setup of the conda environment and a general description of the program structure and command line arguments. Then steps to reproduce the results.


## Reproducing the trained models
To make sure that all the package versions match between the environment that produced the results, install the conda environment listed in `#env`. If exact reproducability is less important, an other option is to install the conda environment in `environment-cross-platform.yml`. 

If everything is setup, run the commands that corresponds to the experiment you wish to produce. The command line agument defaults that are used (which can also be found in `main.py`) are:

`--hidden_size 128 --static_size 2 --dynamic_size 1 --num_layers 1 --dropout 0.1 --train_size 128 --line_unit_price 1 --station_price 5 --station_num_lim 45 --budget 210 --reward weighted --ses_weight 0 --max_grad_norm 2`

The options `var_lambda` and `ggi_weight` were not part of the experiments. The option `--groups_file` can be omited to select the default file (which is differently named between different environments; see `environments/`). An additional analysis method that is available is to plot the best line every $z$ epochs using the `--plot_every z` option. The resulting plots can be found in `result_dir/plots` of the corresponding experiment.

### Simple actor experiments
Run the following command for $mlp, cnn, rnn, pointer \in \text{ARCH}$ and some choice for $x$ and $y$ (the pointer network does not use these variables).

`python main.py --actor_lr 0.001 --critic_lr 0.001 --arch ARCH --environment diagonal_5x5 --epoch_max 200 --actor_mlp_layers x --critic_mlp_layers y  --test`

### Actor attention experiments
Run the following command for $mlp\text{-}att, rnn\text{-}att, pointer \in \text{ARCH}$. Both the MLP-ATT and RNN-ATT used 5 and 4 fully connected layers for the actor and critic networks respectively (this was accidentily left out of the report).

`python main.py --actor_lr 0.01 --critic_lr 0.01 --arch ARCH --environment xian --epoch_max 4000 --actor_mlp_layers 5 --critic_mlp_layers 4 --test`

### Reward function experiments
Run the following commands to reproduce the base model:

`python main.py --actor_lr 0.01 --critic_lr 0.01 --arch pointer --environment diamond_5x5 --epoch_max 500`,

and to reproduce the rationality rewarding models, run:

`python main.py --actor_lr 0.01 --critic_lr 0.01 --arch pointer --environment diamond_5x5 --epoch_max 500 --constraint_free --cf_reward_scaling FN`, 

for $linear, parabolic \in \text{FN}$.

## Evaluating the trained models
The `evaluate.ipynb` contains code to produce average reward tables and plots for the obtained runs. The `--test` command line arguments makes sure that a trained model is also evaluated. This means that the maximum reward (also per group; `result_metric.json`), the loss curve (`loss.png`) and the average generated line (`average_generated_line.png` and `tour_idx_multiple.txt`) are directly available in the `result_dir`. 

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