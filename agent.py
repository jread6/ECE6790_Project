import gymnasium
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from joblib import dump, load
import numpy as np
from filterpy.kalman import KalmanFilter
import time
import warnings
warnings.filterwarnings('ignore')
import pickle



class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        # self.fc = nn.Sequential(
        #     nn.Linear(input_shape, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, n_actions)
        # )
        self.fc = nn.Sequential(
            nn.Linear(input_shape, 100),
            nn.ReLU(),
            nn.Linear(100, n_actions)
        )
        self.activations = torch.empty((0,100), device='cuda')

    def forward(self, x, save_activations=False):
        x = x.float()
        if save_activations:
            for layer in self.fc:
                x = layer(x)
                if isinstance(layer, nn.ReLU):
                    self.activations=torch.cat((self.activations, x), 0)
                    activations = x
        else:
            x = self.fc(x)
        return x , activations

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
    def __init__(self, env, gamma=0.99, eps_start=0.5, eps_end=0.01, eps_decay=0.995, lr=1e-3, memory_size=10000, batch_size=1, target_update=10):
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

        self.policy_net = DQN(env.observation_space.shape[0]*env.observation_space.shape[1], env.action_space.n).to(self.device)
        self.target_net = DQN(env.observation_space.shape[0]*env.observation_space.shape[1], env.action_space.n).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()
        #self.loss_fn = nn.MSELoss()

    def select_action(self, state, eps, save_activations=False):
        if random.random() > eps:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device)
                state = state.unsqueeze(0)
                
                q_values, activations = self.policy_net(state.reshape((state.shape[0], state.shape[1]*state.shape[2])), save_activations=save_activations)
                action = q_values.argmax().item()
        else:
            action = self.env.action_space.sample()
        return action, activations

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.tensor(states, device=self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)
        next_states = torch.tensor(next_states, device=self.device)
        dones = torch.tensor(dones, device=self.device)

        
        q_values = self.policy_net(states.reshape((states.shape[0], states.shape[1]*states.shape[2])))
        q_values = q_values[range(len(actions.long())), actions.long()]
        
        next_q_values = self.target_net(next_states.reshape((next_states.shape[0], next_states.shape[1]*next_states.shape[2]))).max(1)[0].contiguous().detach()
        expected_q_values = rewards + self.gamma * next_q_values * (1-dones.to(torch.int))
        
        loss = self.loss_fn(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, num_episodes):
        eps = self.eps_start
        rewards = []
        self.env.reset()
        
        for episode in range(num_episodes):
            state, goal_pos = self.env.reset()
            done = False
            episode_reward = 0
            iter = 0

            while not done:
                self.env.render()
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
            print(f"Episode: {episode+1}, reward: {episode_reward:.2f}, eps: {eps:.2f}, steps: {self.steps}")

        return rewards

    def test(self, num_episodes):
        rewards = []
        goal_positions = np.empty((0,3))
        for episode in range(num_episodes):
            state, goal_pos = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                goal_positions = np.append(goal_positions, np.array([[episode+1, goal_pos[0], goal_pos[1]]]), axis=0)
                self.env.render()
                action, _ = self.select_action(state, 0, save_activations=True)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                episode_reward += reward
            rewards.append(episode_reward)
            print(f"Episode: {episode+1}, reward: {episode_reward:.2f}")

        return rewards, goal_positions
    
    def kalman(self, num_episodes):
        rewards = []
        goal_positions = np.empty((0,2))
        fa_loaded = load('fa_model.joblib')
        factors = np.empty((0, 6))
        # define Kalman filter
        kf = KalmanFilter(dim_x=6, dim_z=2)

        # define state transition matrix
        A = np.array([[1., 0., 1., 0., 0.5, 0.],
                      [0., 1., 0., 1., 0., 0.5],
                      [0., 0., 1., 0., 1., 0.],
                      [0., 0., 0., 1., 0., 1.],
                      [0., 0., 0., 0., 1., 0.],
                      [0., 0., 0., 0., 0., 1.]])

        kf.F = A

        # define observation matrix
        H = np.array([[1., 0., 0., 0., 0., 0.],
                      [0., 1., 0., 0., 0., 0.]])

        kf.H = H

        # define measurement noise covariance matrix
        R = np.array([[0.1, 0.],
                      [0., 0.1]])

        kf.R = R

        # define initial state vector
        x = np.array([[0., 0., 0., 0., 0., 0.]]).T

        kf.x = x

        # define initial state covariance matrix
        P = np.array([[1., 0., 0., 0., 0., 0.],
                      [0., 1., 0., 0., 0., 0.],
                      [0., 0., 1., 0., 0., 0.],
                      [0., 0., 0., 1., 0., 0.],
                      [0., 0., 0., 0., 1., 0.],
                      [0., 0., 0., 0., 0., 1.]])

        kf.P = P
        
        with open('perceptron_model_action.pkl', 'rb') as file:
            model_action = pickle.load(file)
        

        for episode in range(num_episodes):
            state, goal_pos = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                self.env.render()
                action, activations = self.select_action(state, 0, save_activations=True)
                numpy_activations = activations.cpu().numpy()
                new_factors = fa_loaded.transform(numpy_activations)
                predicted_action = model_action.predict(new_factors)
                predicted_action = round(predicted_action[0])
                print(predicted_action,action)
                factors = np.vstack([factors, new_factors])
                goal_positions = np.append(goal_positions, np.array([[episode+1, action]]), axis=0)
                next_state, reward, done, _ = self.env.step(predicted_action)
                state = next_state
                episode_reward += reward
           
            rewards.append(episode_reward)
            #print(f"Episode: {episode+1}, reward: {episode_reward:.2f}")

        return rewards, goal_positions, factors
    