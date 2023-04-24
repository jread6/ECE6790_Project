import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pyglet

class GridWorldEnv(gym.Env):
    def __init__(self, size=5):
        self.size = size
        self.grid = np.zeros((self.size, self.size))
        self.agent_pos = [int(self.size/2), int(self.size/2)]
        self.potential_goal_positions = [[0, int(self.size/2)], [int(self.size/4), int(self.size/4)], [int(self.size/4), int(3*self.size/4)], [int(self.size/2), 0], [int(self.size/2), self.size-1], [int(3*self.size/4), int(self.size/4)], [int(3*self.size/4), int(3*self.size/4)], [self.size-1, int(self.size/2)]]
        self.goal_pos = random.choice(self.potential_goal_positions)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.size, self.size), dtype=np.uint8)
        self.window = pyglet.window.Window(self.size * 50, self.size * 50)

        @self.window.event
        def on_draw():
            self.window.clear()
            for i in range(self.size):
                for j in range(self.size):
                    x = i * 50
                    y = j * 50
                    pyglet.graphics.draw(4, pyglet.gl.GL_LINE_LOOP,
                                         ('v2i', (x, y, x + 50, y, x + 50, y + 50, x, y + 50)))
            for goal in self.potential_goal_positions:
                goal_x = goal[0] * 50 + 25
                goal_y = goal[1] * 50 + 25
                if goal == self.goal_pos:
                    pyglet.graphics.draw(4, pyglet.gl.GL_QUADS,
                                         ('v2i', (goal_x - 10, goal_y - 10,
                                                  goal_x + 10, goal_y - 10,
                                                  goal_x + 10, goal_y + 10,
                                                  goal_x - 10, goal_y + 10)),
                                         ('c3B', (128,128,128)*4))
                else:
                    pyglet.graphics.draw(4, pyglet.gl.GL_LINE_LOOP,
                                         ('v2i', (goal_x - 10, goal_y - 10,
                                                  goal_x + 10, goal_y - 10,
                                                  goal_x + 10, goal_y + 10,
                                                  goal_x - 10, goal_y + 10)),
                                         ('c3B', (128,128,128)*4))
            agent_x = self.agent_pos[0] * 50
            agent_y = self.agent_pos[1] * 50
            pyglet.graphics.draw(4, pyglet.gl.GL_QUADS,
                                    ('v2i', (agent_x, agent_y, agent_x + 50, agent_y,
                                            agent_x + 50, agent_y + 50, agent_x, agent_y + 50)))
            
    def step(self, action):
        prev_pos = self.agent_pos
        if action == 0: # up
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1: # right
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        elif action == 2: # down
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)
        else: # left
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)

        done = (self.agent_pos == self.goal_pos)
        # reward = 1 if done else -1
        
        ##Uncomment modified rewards based on direction of motion below
        # reward = 1 if self.distance(self.goal_pos,self.agent_pos)<self.distance(self.goal_pos,prev_pos) else -1
        # reward = float(1/((self.distance(self.goal_pos,self.agent_pos)-self.distance(self.goal_pos,prev_pos))+1e-3))
        reward = 10 if done else 1 - self.distance(self.goal_pos,self.agent_pos)
        observation = self._get_observation()
        return observation, reward, done, {}
    
    def distance(self, pos1, pos2):
        return ((pos1[0]-pos2[0])**2+(pos1[1]-pos2[1])**2)**0.5
    
    def reset(self):
        self.agent_pos = [int(self.size/2), int(self.size/2)]
        self.goal_pos = random.choice(self.potential_goal_positions)
        return self._get_observation(), self.goal_pos
    
    def render(self):
        self.window.switch_to()
        self.window.dispatch_events()
        self.window.dispatch_event('on_draw')
        self.window.flip()

    def _get_observation(self):
        observation = np.zeros((self.size, self.size))
        observation[tuple(self.agent_pos)] = 1
        observation[tuple(self.goal_pos)] = -1
        return observation
