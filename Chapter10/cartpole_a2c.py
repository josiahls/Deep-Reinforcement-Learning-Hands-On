# from torch.utils.tensorboard import SummaryWriter
from itertools import count

import torch.nn.functional as F
import numpy as np
import torch as T
import gym
import torch.nn as nn
import torch.optim as optim


class GenericNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(GenericNetwork, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, observation):
        state = T.FloatTensor(observation).to(self.device) if isinstance(observation,np.ndarray) else observation
        output = F.relu(self.fc1(state))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output



def choose_action(network, state):
    probabilities = F.softmax(network.forward(state))
    action_probs = T.distributions.Categorical(probabilities)
    action = action_probs.sample()

    return action.item()


if __name__ == "__main__":

    env = gym.make("CartPole-v1")
    # writer = SummaryWriter("run")

    device = T.device("cuda" if T.cuda.is_available() else "cpu")

    actor = GenericNetwork(0.0001, 4, 32, 32, 2)
    critic = GenericNetwork(0.005, 4, 32, 2, 1)

    eps_states, eps_actions, eps_rewards = [], [], []

    gamma = 0.99
    scores = []

    for episode in count():
        score = 0
        q_val = 0
        done = False
        state = env.reset()

        while not done:
            action = choose_action(actor, state)
            state_, reward, done, _ = env.step(action)

            eps_actions.append(int(action))
            eps_states.append(state)
            eps_rewards.append(reward)

            state = state_
            score += reward

            if done:
                R = 0
                len_episode = len(eps_actions)

                actor.optimizer.zero_grad()
                critic.optimizer.zero_grad()

                eps_states_ts = T.FloatTensor(eps_states).to(device=device)
                eps_actions_ts = T.LongTensor(eps_actions).to(device=device)
                eps_rewards_ts = T.FloatTensor(eps_rewards).to(device=device)

                for i in range(len_episode):
                    time = len_episode - i - 1

                    R *= gamma
                    R += eps_rewards[time]

                    critic_value_time_ts = critic.forward(eps_states_ts[time]).squeeze()

                    delta = R - critic_value_time_ts

                    logits = actor(eps_states_ts[time])
                    log_prob = F.log_softmax(logits, dim=0)
                    log_prob_actions = log_prob[eps_actions_ts[time]] * delta

                    actor_loss = -log_prob_actions
                    critic_loss = (delta ** 2)

                    (actor_loss + critic_loss).backward()

                    actor.optimizer.step()
                    critic.optimizer.step()

                eps_states.clear()
                eps_actions.clear()
                eps_rewards.clear()

        scores.append(score)
        mean_score = np.array(scores[-100:])
        mean_score = np.mean(mean_score)

        # writer.add_scalar("score", score, episode)
        # writer.add_scalar("mean_score", mean_score, episode)

        if episode % 5 == 0:
            print("episode :%d, score :%.3f, mean_score :%.3f" % (episode, score, mean_score))
