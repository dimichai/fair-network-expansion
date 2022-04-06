import datetime
import os
from pathlib import Path
import time
import torch
import torch.optim as optim
from actor import DRL4Metro
from constraints import Constraints
from critic import StateCritic
import constants
from reward import od_utility
import matplotlib.pyplot as plt

device = constants.device


def update_dynamic(dynamic, current_selected_index):
    """Updates the dynamic representation of the actor to a sparse matrix with all so-far selected stations.
    Note: this does not seem to be correct. The current implementation of metro expansion does not have any 'dynamic' elements, like demand.

    Args:
        dynamic (torch.Tensor): the current dynamic matrix.
        current_selected_index (np.int64): the latest selected station.

    Returns:
        torch.Tensor: the new dynamic matrix, where all selected stations are assigned 1.
    """
    dynamic = dynamic.clone()
    dynamic[0, 0, current_selected_index] = float(1)

    return dynamic


class Trainer(object):
    """Responsible for the wholet raining process."""

    def __init__(self, environment, constraints: Constraints, args):
        super(Trainer, self).__init__()

        # Prepare the models
        # TODO: replace allowed_next_vector_indices with environment.constraints... whatever
        self.environment = environment
        self.actor = DRL4Metro(args.static_size,
                          args.dynamic_size,
                          args.hidden_size,
                          args.num_layers,
                          args.dropout,
                          update_dynamic,
                          environment.update_mask,
                          v_to_g_fn=environment.vector_to_grid,
                          vector_allow_fn=constraints.allowed_vector_indices).to(device)

        self.critic = StateCritic(args.static_size, args.dynamic_size,
                             args.hidden_size, environment.grid_size).to(device)

    def train(self, args):
        if args.checkpoint:
            self.actor.load_state_dict(torch.load(args.checkpoint / 'actor.pt', device))
            self.critic.load_state_dict(torch.load(args.checkpoint / 'critic.pt', device))

        now = datetime.datetime.today().strftime('%Y%m%d_%H_%M_%S.%f')
        save_dir = args.result_path / now

        print(f'Starts training on {device} - Model location is {save_dir}')

        checkpoint_dir = save_dir / 'checkpoints'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        actor_optim = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        critic_optim = optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        average_reward_list, actor_loss_list, critic_loss_list, average_od_list, average_Ac_list = [], [], [], [], []
        # best_params = None
        best_reward = 0

        static = self.environment.static
        dynamic = torch.zeros((1, args.dynamic_size, self.environment.grid_size),
                            device=device).float()  # size with batch

        for epoch in range(args.epoch_max):
            self.actor.train()
            self.critic.train()

            epoch_start = time.time()
            od_list, social_equity_list = [], []
            actor_loss = critic_loss = rewards_sum = 0

            for _ in range(args.train_size):  # this loop accumulates a batch
                tour_idx, tour_logp = self.actor(static, dynamic, args.station_num_lim, budget=args.budget,
                                            line_unit_price=args.line_unit_price, station_price=args.station_price,
                                            decoder_input=None, last_hh=None)

                # TODO: add different conditions for calculating the reward function.
                reward = od_utility(tour_idx, self.environment)
                od_list.append(reward.item())
                social_equity_list.append(0)

                critic_est = self.critic(static, dynamic, args.hidden_size,
                                    self.environment.grid_x_size, self.environment.grid_y_size).view(-1)
                advantage = (reward - critic_est)
                per_actor_loss = -advantage.detach() * tour_logp.sum(dim=1)
                per_critic_loss = advantage ** 2

                actor_loss += per_actor_loss
                critic_loss += per_critic_loss
                rewards_sum += reward

            actor_loss = actor_loss / args.train_size
            critic_loss = critic_loss / args.train_size
            avg_reward = rewards_sum / args.train_size
            average_od = sum(od_list)/len(od_list)
            average_Ac = sum(social_equity_list)/len(social_equity_list)

            average_reward_list.append(avg_reward.half().item())
            actor_loss_list.append(actor_loss.half().item())
            critic_loss_list.append(critic_loss.half().item())
            average_od_list.append(average_od)
            average_Ac_list.append(average_Ac)

            actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), args.max_grad_norm)
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), args.max_grad_norm)
            critic_optim.step()

            cost_time = time.time() - epoch_start
            print('epoch %d, average_reward: %2.3f, actor_loss: %2.4f,  critic_loss: %2.4f, cost_time: %2.4fs'
                % (epoch, avg_reward.item(), actor_loss.item(), critic_loss.item(), cost_time))

            torch.cuda.empty_cache()  # reduce memory

            # Save the weights of an epoch
            epoch_dir = checkpoint_dir / str(epoch)
            if not os.path.exists(epoch_dir):
                os.makedirs(epoch_dir)

            torch.save(self.actor.state_dict(), epoch_dir / 'actor.pt')
            torch.save(self.critic.state_dict(), epoch_dir / 'critic.pt')

            # Save best model parameters
            if avg_reward.item() > best_reward:
                best_reward = avg_reward.item()

                torch.save(self.actor.state_dict(), save_dir / 'actor.pt')
                torch.save(self.critic.state_dict(), save_dir / 'critic.pt')

        with open(save_dir / 'reward_actloss_criloss.txt', 'w') as f:
            for i in range(args.epoch_max):
                per_average_reward_record = average_reward_list[i]
                per_actor_loss_record = actor_loss_list[i]
                per_critic_loss_record = critic_loss_list[i]
                per_epoch_od = average_od_list[i]
                per_epoch_Ac = average_Ac_list[i]

                to_write = f'{per_average_reward_record}\t{per_actor_loss_record}\t{per_critic_loss_record}\t{per_epoch_od}\t{per_epoch_Ac}\n'

                f.write(to_write)

        plt.plot(average_reward_list, '-', label="reward")
        plt.title(f'Reward vs. epochs - {now}')
        plt.ylabel('Reward')
        plt.legend(loc='best')
        plt.savefig(save_dir / 'loss.png', dpi=800)
