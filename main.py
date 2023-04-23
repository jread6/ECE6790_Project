from game import GridWorldEnv
from agent import Agent
from matplotlib import pyplot as plt
import torch
import numpy as np
NUM_EPISODES = 500
EPS_MIN = 0.01

def main():
    size = 9
    env = GridWorldEnv(size = size)

    load_weights = False
    train = False
    test = False
    kalman = True

    network_type = 'small'

    # Train for 100 episodes
    if train:
        agent = Agent(env, eps_start=0.75, eps_decay=0.995, batch_size=128)

        if load_weights:
            # Load the saved state dictionary
            state_dict = torch.load('policy_net_weights_'+str(network_type)+'.pth')
            
            # Update the agent's policy network parameters
            agent.policy_net.load_state_dict(state_dict)

        rewards_vs_episodes = agent.train(NUM_EPISODES)

        np.savetxt('rewards_vs_episodes_'+str(network_type)+'.csv', rewards_vs_episodes, delimiter=',')

        # Save the state dictionary of the agent's policy network
        torch.save(agent.policy_net.state_dict(), 'policy_net_weights_'+str(network_type)+'_train.pth')

        plt.plot(rewards_vs_episodes)
        plt.show()
    if test:
        agent = Agent(env, batch_size=1)
        # Load the saved state dictionary
        state_dict = torch.load('policy_net_weights_'+str(network_type)+'_train.pth')
        
        # Update the agent's policy network parameters
        agent.policy_net.load_state_dict(state_dict)

        num_trials=20000
        rewards_vs_episodes, goal_positions = agent.test(num_trials)
        # # write results to a file
        activations = agent.policy_net.activations.cpu().numpy()
        np.savetxt('data/network_activations_'+str(network_type)+'_'+str(num_trials)+'_trials.csv', activations, delimiter=',')

        # np.savetxt('rewards_vs_episodes_'+str(network_type)+'_'+str(num_trials)+'_trials.csv', rewards_vs_episodes, delimiter=',')
        # np.savetxt('goal_positions_'+str(network_type)+'_'+str(num_trials)+'_trials.csv', goal_positions, delimiter=',')
        
    if kalman:
        print("Kalman")
        agent = Agent(env, batch_size=1)
        # Load the saved state dictionary
        state_dict = torch.load('policy_net_weights_'+str(network_type)+'_train.pth')
        
        # Update the agent's policy network parameters
        agent.policy_net.load_state_dict(state_dict)

        num_trials=5000
        rewards_vs_episodes, goal_positions, factors = agent.kalman(num_trials)
        np.savetxt('positions.csv', goal_positions, delimiter=',')
        np.savetxt('factors.csv', factors, delimiter=',')

if __name__ == "__main__":
    main()