from game import GridWorldEnv
from agent import Agent

NUM_EPISODES = 100
EPS_MIN = 0.01

def main():
    env = GridWorldEnv()
    agent = Agent(env)

    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Choose action
            action = agent.select_action(state, eps=agent.eps_end)

            # Take step
            next_state, reward, done, _ = env.step(action)

            # Add to replay buffer
            agent.memory.push(state, action, reward, next_state, done)

            # Update state and total reward
            state = next_state
            total_reward += reward

            # Optimize model
            agent.optimize_model()

            # Update target network
            if agent.steps % agent.target_update == 0:
                agent.target_net.load_state_dict(agent.policy_net.state_dict())

            # Decay epsilon
            agent.eps_end = max(agent.eps_end * agent.eps_decay, EPS_MIN)

            # Increment step counter
            agent.steps += 1

        # Print episode statistics
        print(f"Episode {episode + 1}, total reward: {total_reward}")
