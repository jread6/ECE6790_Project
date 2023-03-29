import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

# chatGPT implementation. Currently has some bugs

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        x = x.float()
        x = self.fc(x)
        return x

class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class Agent():
    def __init__(self, env, gamma=0.99, eps_start=1.0, eps_end=0.01, eps_decay=0.9995, lr=1e-3, memory_size=10000, batch_size=128, target_update=10):
        self.env = env
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.lr = lr
        self.memory = ReplayBuffer(memory_size)
        self.batch_size = batch_size
        self.target_update = target_update
        self.steps = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DQN(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.target_net = DQN(env.observation_space.shape[0], env.action_space.n).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, state, eps):
        if random.random() > eps:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device)
                state = state.unsqueeze(0)
                q_values = self.policy_net(state)
                action = q_values.argmax().item()
        else:
            action = self.env.action_space.sample()
        return action

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.tensor(states, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)
        next_states = torch.tensor(next_states, device=self.device)
        dones = torch.tensor(dones, device=self.device)

        print(self.policy_net(states).shape, actions.shape)
        q_values = self.policy_net(states)
        q_values = q_values[range(len(actions)), actions]
        next_q_values = self.target_net(next_states).max(1)[0].contiguous().detach()
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)
        loss = self.loss_fn(q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, num_episodes):
        eps = self.eps_start
        rewards = []
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.select_action(state, eps)
                next_state, reward, done, _ = self.env.step(action)
                self.memory.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                self.steps += 1

                if self.steps % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                self.optimize_model()

            rewards.append(episode_reward)
            eps = max(self.eps_end, eps * self.eps_decay)
            print(f"Episode: {episode+1}, reward: {episode_reward:.2f}, eps: {eps:.2f}")

        return rewards
