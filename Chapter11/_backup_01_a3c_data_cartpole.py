#!/usr/bin/env python3
import gym
import ptan
import numpy as np
import argparse
import collections
# from tensorboardX import SummaryWriter
import torch.nn as nn
import torch
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import time
import sys
from basic_agents import *

class RewardTracker:
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):pass
        # self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        # print(reward)
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards), mean_reward, speed, epsilon_str
        ))
        sys.stdout.flush()
        # if epsilon is not None:
        #     self.writer.add_scalar("epsilon", epsilon, frame)
        # self.writer.add_scalar("speed", speed, frame)
        # self.writer.add_scalar("reward_100", mean_reward, frame)
        # self.writer.add_scalar("reward", reward, frame)
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False

# from lib import common
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

    def forward(self, x):
        fx=x.float()
        return self.policy(fx),self.value(fx)


GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128

REWARD_STEPS = 4
CLIP_GRAD = 0.1

PROCESSES_COUNT = 1 # 4
NUM_ENVS = 1 # 15

# if True:
#     ENV_NAME = "PongNoFrameskip-v4"
#     NAME = 'pong'
#     REWARD_BOUND = 18
# else:
#     ENV_NAME = "BreakoutNoFrameskip-v4"
#     NAME = "breakout"
#     REWARD_BOUND = 400
#
if True:
    ENV_NAME = "CartPole-v1"
    NAME = 'cartpole'
    REWARD_BOUND = 200
else:
    ENV_NAME = "CartPole-v1"
    NAME = "cartpole2"
    REWARD_BOUND = 200



# def make_env():
#     return ptan.common.wrappers.wrap_dqn(gym.make(ENV_NAME))

def make_env():
    _env=gym.make(ENV_NAME)
    # _env.reset()
    return _env # common.PixelObservationWrapper(_env,boxify=True)


TotalReward = collections.namedtuple('TotalReward', field_names='reward')


class Debug(ActorCriticAgent):
    def __call__(self, *args, **kwargs):
        print( *args, **kwargs)
        return super().__call__( *args, **kwargs)


def data_func(net, device, train_queue):
    envs = [make_env() for _ in range(NUM_ENVS)]
    # agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], device=device, apply_softmax=True)
    agent=ActorCriticAgent(net,device='cuda')
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    for exp in exp_source:
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            # print(exp.last_state is None,len(new_rewards),flush=True)
            train_queue.put(TotalReward(reward=np.mean(new_rewards)))
        train_queue.put(exp)


def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print(' - shape: ',tensor.shape)
                print()
            except AttributeError:
                getBack(n[0])


def r_estimate(s, r, d_mask,non_d_mask, model, val_gamma, device):
    "Returns rewards `r` estimated direction by `model` from states `s`"
    r_np = np.array(r, dtype=np.float32)
    # print(r_np[d_mask].mean(), r_np[non_d_mask].mean())
    #     print(len(d_mask),len(r),len(s))
    if len(d_mask) != 0:
        s_v = torch.FloatTensor(s).to(device)
        v = model(s_v)[1]  # Remember that models are going to return the actions and the values
        v_np = v.data.cpu().numpy()[:, 0]
        r_np[d_mask] += val_gamma * v_np
    return r_np

ExperienceFirstLastNew = collections.namedtuple('ExperienceFirstLastNew', ('s', 'a', 'r', 'sp','d'))

def unbatch(batch, net, last_val_gamma, device='cpu'):
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
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if not exp.done: #exp.last_state is not None:
            not_done_idx.append(idx)
            # if exp.last_state is None:print(exp)
            last_states.append(np.array(exp.last_state, copy=False))
        # else:
        #     print(exp,'is done, so skipping')
    states_v = torch.FloatTensor(states).to(device)
    actions_t = torch.LongTensor(actions).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        # print(last_states)
        last_states_v = torch.FloatTensor(last_states).to(device)
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        # print(last_vals_v.data.cpu().numpy().mean())
        rewards_np[not_done_idx] += last_val_gamma * last_vals_np

    # print(len(not_done_idx),len(rewards_np),len(last_states))
    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_t, ref_vals_v



def getModelconf(model,do_grads=False):
    for p in model.parameters():
        if not do_grads:print('Shape: ',p.shape, 'P mean: ',p.mean(), '\nP Max: ',p.max(),'P Min: ',p.min())
        else:           print('Shape: ',p.shape, 'P mean: ',p.grad.mean(), '\nP Max: ',p.grad.max(),'P Min: ',p.grad.min())

if __name__ == "__main__":
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=False,default='cartpole', help="Name of the run")
    args = parser.parse_args()
    device = "cuda" # if args.cuda else "cpu"

    # writer = SummaryWriter(comment="-a3c-data_" + NAME + "_" + args.name)

    env = make_env()
    net = LinearA2C(env.observation_space.shape, env.action_space.n).to(device)
    net.share_memory()

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    train_queue = mp.Queue(maxsize=PROCESSES_COUNT)
    data_proc_list = []
    for _ in range(PROCESSES_COUNT):
        data_proc = mp.Process(target=data_func, args=(net, device, train_queue))
        data_proc.start()
        data_proc_list.append(data_proc)

    batch = []
    step_idx = 0
    batch_num = 0
    try:
        with RewardTracker(None, stop_reward=REWARD_BOUND) as tracker:
            # with ptan.common.utils.TBMeanTracker(None, batch_size=100) as tb_tracker:
            while True:
                train_entry = train_queue.get()
                # print(train_entry)
                if isinstance(train_entry, TotalReward):
                    if tracker.reward(train_entry.reward, step_idx):
                        break
                    continue

                step_idx += 1
                # print(train_entry)
                batch.append(train_entry)
                if len(batch) < BATCH_SIZE:
                    continue
                batch_num+=1
                # states_v1, actions_t, vals_ref_v1 = \
                #     common.unpack_batch(batch, net, last_val_gamma=GAMMA**REWARD_STEPS, device=device)
                # print(batch)
                states_v, actions_t, vals_ref_v = \
                    unbatch(batch, net, last_val_gamma=GAMMA**REWARD_STEPS, device=device)
                print(np.array([_b.reward for _b in batch]).mean(), vals_ref_v.float().mean())
                # print(vals_ref_v.mean(), np.mean([o.reward for o in batch]))
                # print(states_v.shape,actions_t.shape,vals_ref_v.shape)
                # vals_ref_v=vals_ref_v.squeeze(1)
                # print(states_v.float().mean(),actions_t.float().mean(),vals_ref_v.float().mean())
                batch.clear()
                print(batch_num)

                optimizer.zero_grad()
                logits_v, value_v = net(states_v)
                # print(logits_v.shape,value_v.shape)

                """
                <TBackward object at 0x7f4a0039c908><AccumulateGrad object at 0x7f4a0039c9e8>
                Jumps to  -log_prob_actions_v.mean()
                <TBackward object at 0x7f4a0039c828><AccumulateGrad object at 0x7f4a0039c8d0> vals_ref_v
                Jumps to the NN Module for value, go to module.value:
                <SqueezeBackward1 object at 0x7f4a0039c6d8> value_v.squeeze(-1)
                <MseLossBackward object at 0x7f4a0039c630> F.mse_loss
                <TBackward object at 0x7f4a0039c908><AccumulateGrad object at 0x7f4a0039c9e8> loss_value_v
                """
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                """
                Jumps to the NN Module for policy, go to module.policy (again):
                <LogSoftmaxBackward object at 0x7f4a0039c748>F.log_softmax(logits_v, dim=1)
                """
                """
                Jumps to F.softmax(logits_v, dim=1)
                Jumps to the NN Module for policy, go to module.policy (again):
                <LogSoftmaxBackward object at 0x7f4a0039c828> F.log_softmax(logits_v, dim=1)
                <TBackward object at 0x7f4a0039c978><AccumulateGrad object at 0x7f4a0039ca58>log_prob_v
                """
                log_prob_v = F.log_softmax(logits_v, dim=1)
                adv_v = vals_ref_v - value_v.detach()

                """
                Jumps to  F.log_softmax(logits_v, dim=1)
                <IndexBackward object at 0x7f4a0039c6a0> log_prob_v[range(BATCH_SIZE), actions_t]
                <MulBackward0 object at 0x7f4a0039c5f8>  adv_v *
                """
                # print(log_prob_v.shape)
                log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions_t]
                """
                <MeanBackward0 object at 0x7f4a0039c630> log_prob_actions_v.mean()
                <NegBackward object at 0x7f4a0039c240> -log_prob_actions_v
                """
                loss_policy_v = -log_prob_actions_v.mean()

                """
                Jumps to log_prob_v F.log_softmax(logits_v, dim=1)
                Jumps to the NN Module for policy, go to module.policy:
                <SoftmaxBackward object at 0x7f1159e3c7f0>
                """
                prob_v = F.softmax(logits_v, dim=1)
                """ 
                Jumps to F.mse_loss(value_v.squeeze(-1), vals_ref_v)
                <MulBackward0 object at 0x7f1159e3c780> prob_v * log_prob_v
                <SumBackward1 object at 0x7f1159e3c710>   .sum(dim=1)
                <MeanBackward0 object at 0x7f1159e3c6a0>  .mean()
                <MulBackward0 object at 0x7f1159e3c5f8>   ENTROPY_BETA * 
                """
                entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

                """ 
                <AddBackward0 object at 0x7f1159e3c400> entropy_loss_v
                <AddBackward0 object at 0x7f1159e3c588> loss_value_v +    loss_policy_v
                """
                loss_v = entropy_loss_v + loss_value_v + loss_policy_v

                # print(entropy_loss_v,loss_policy_v,loss_value_v)
                loss_v.backward()
                # getBack(loss_v.grad_fn)
                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                # print(step_idx)
                optimizer.step()
                # if batch_num>1:
                #     getModelconf(net,True)
                #     if batch_num>2:raise Exception
                print(batch_num,loss_v.detach(),entropy_loss_v.detach(),loss_value_v.detach(),loss_policy_v.detach())

                    # tb_tracker.track("advantage", adv_v, step_idx)
                    # tb_tracker.track("values", value_v, step_idx)
                    # tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                    # tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                    # tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                    # tb_tracker.track("loss_value", loss_value_v, step_idx)
                    # tb_tracker.track("loss_total", loss_v, step_idx)
    finally:
        for p in data_proc_list:
            p.terminate()
            p.join()
