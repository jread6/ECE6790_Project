from game import GridWorldEnv
from agent import Agent
from matplotlib import pyplot as plt
NUM_EPISODES = 1000
EPS_MIN = 0.01

def main():
    env = GridWorldEnv()
    agent = Agent(env)
    rewards_vs_episodes = agent.train(NUM_EPISODES)
    plt.plot(rewards_vs_episodes)
    plt.show()
    #assert 1==0

    # for episode in range(NUM_EPISODES):
        # state = env.reset()
        # total_reward = 0
        # done = False
        
        # #env.render()

        # while not done:
            # # Choose action
            # action = agent.select_action(state, eps=agent.eps_start)
            # #env.step(action)

            # # Take step
            # next_state, reward, done, _ = env.step(action)

            # # Add to replay buffer
            # agent.memory.push(state, action, reward, next_state, done)

            # # Update state and total reward
            # state = next_state
            # total_reward += reward

            # # Optimize model
            # agent.optimize_model()

            # # Update target network
            # if agent.steps % agent.target_update == 0:
                # agent.target_net.load_state_dict(agent.policy_net.state_dict())

            # # Decay epsilon
            # agent.eps_start = max(agent.eps_start * agent.eps_decay, EPS_MIN)

            # # Increment step counter
            # agent.steps += 1
        # #env.reset()

        # # Print episode statistics
        # print(f"Episode {episode + 1}, total reward: {total_reward}")

if __name__ == "__main__":
    main()