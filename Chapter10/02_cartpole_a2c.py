#!/usr/bin/env python3
import gym
import ptan
import numpy as np
import argparse
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim

from lib import common

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128
NUM_ENVS = 15

REWARD_STEPS = 4
CLIP_GRAD = 0.1


class LinearA2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(LinearA2C, self).__init__()

        self.policy = nn.Sequential(
            nn.Linear(input_shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(input_shape[0], 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o=self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx=x.float()
        return self.policy(fx),self.value(fx)


def unpack_batch(batch, net, device='cpu'):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        # print(exp.state.shape,exp.action,exp.reward,exp.last_state.shape if exp.last_state is not None else None)
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))
    states_v = torch.FloatTensor(states).to(device)
    actions_t = torch.LongTensor(actions).to(device)
    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = torch.FloatTensor(last_states).to(device)
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += GAMMA ** REWARD_STEPS * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_t, ref_vals_v


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", default='silly',help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    make_env = lambda: gym.make("CartPole-v1")

    # env = make_env()
    envs = [make_env() for _ in range(NUM_ENVS)]
    writer = SummaryWriter(comment="-pong-a2c_" + args.name)

    net = LinearA2C(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
    print('model start',np.mean([o.detach().cpu().numpy().mean() for o in net.parameters()]))

    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    batch = []
    batch_n=0

    with common.RewardTracker(writer, stop_reward=200) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):
                batch.append(exp)

                # handle new rewards
                new_rewards = exp_source.pop_total_rewards()
                if new_rewards:
                    if tracker.reward(new_rewards[0], step_idx):
                        break

                if len(batch) < BATCH_SIZE:
                    continue

                print(batch_n)
                batch_n+=1

                states_v, actions_t, vals_ref_v = unpack_batch(batch, net, device=device)
                batch.clear()

                optimizer.zero_grad()
                logits_v, value_v = net(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                log_prob_v = F.log_softmax(logits_v, dim=1)
                # adv_v = vals_ref_v - value_v.detach()
                adv_v = vals_ref_v - value_v.squeeze(-1).detach()
                log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions_t]
                loss_policy_v = -log_prob_actions_v.mean()

                prob_v = F.softmax(logits_v, dim=1)
                entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

                # calculate policy gradients only
                loss_policy_v.backward(retain_graph=True)
                grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                        for p in net.parameters()
                                        if p.grad is not None])

                # apply entropy and value gradients
                loss_v = entropy_loss_v + loss_value_v
                loss_v.backward()
                print('clipping',CLIP_GRAD,
                      np.mean([o.detach().cpu().numpy().mean() for o in net.parameters()]))
                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                optimizer.step()
                # get full loss
                loss_v += loss_policy_v

                print(loss_v.detach())

                # tb_tracker.track("advantage",       adv_v, step_idx)
                # tb_tracker.track("values",          value_v, step_idx)
                # tb_tracker.track("batch_rewards",   vals_ref_v, step_idx)
                # tb_tracker.track("loss_entropy",    entropy_loss_v, step_idx)
                # tb_tracker.track("loss_policy",     loss_policy_v, step_idx)
                # tb_tracker.track("loss_value",      loss_value_v, step_idx)
                # tb_tracker.track("loss_total",      loss_v, step_idx)
                # tb_tracker.track("grad_l2",         np.sqrt(np.mean(np.square(grads))), step_idx)
                # tb_tracker.track("grad_max",        np.max(np.abs(grads)), step_idx)
                # tb_tracker.track("grad_var",        np.var(grads), step_idx)
