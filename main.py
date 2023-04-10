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
    agent = Agent(env, eps_start=0.5, eps_decay=0.995)

    load_weights = False

    if load_weights:
        # Load the saved state dictionary
        state_dict = torch.load('policy_net_weights_E2.pth')
        
        # Update the agent's policy network parameters
        agent.policy_net.load_state_dict(state_dict)

    # Train for 50 episodes
    rewards_vs_episodes = agent.train(NUM_EPISODES)

    np.savetxt('rewards_vs_episodes_Full.csv', rewards_vs_episodes, delimiter=',')

    # Save the state dictionary of the agent's policy network
    torch.save(agent.policy_net.state_dict(), 'policy_net_weights_Full.pth')

    plt.plot(rewards_vs_episodes)
    plt.show()


    # # Load the saved state dictionary
    # state_dict = torch.load('policy_net_weights.pth')
    
    # # Update the agent's policy network parameters
    # agent.policy_net.load_state_dict(state_dict)
    
    # # Traing for 50 more episodes
    # rewards_vs_episodes = agent.train(NUM_EPISODES)
    # plt.plot(rewards_vs_episodes)
    # plt.show()

    # # Save the state dictionary of the agent's policy network
    # torch.save(agent.policy_net.state_dict(), 'policy_net_weights_E2.pth')

if __name__ == "__main__":
    main()