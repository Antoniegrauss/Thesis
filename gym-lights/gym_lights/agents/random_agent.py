import gym
from gym import envs

env = gym.make('gym_lights:lights-2-rooms')
env.reset()
for _ in range(100):
    env.render()
    action = env.action_space.sample()
    env.step(action=action)
    env.render()
