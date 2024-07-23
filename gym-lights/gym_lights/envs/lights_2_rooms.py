import random
from typing import Generic, Optional, SupportsFloat, Tuple, TypeVar, Union
import gym
import numpy as np
from matplotlib import pyplot as plt

from Algorithm import environment_2_rooms


class Lights2RoomsEnv(gym.Env):

    def __init__(self, episode_length=100):
        # Clone environment from classes.py
        self.episode_length = episode_length
        self.configure_environment()
        self.initial_state = tuple(self.state.tolist())
        self.step(action=None)
        self.reset()
        self.ax_env = None

    # Use the do_actions_move_time
    def step(self, action):
        # set reward
        if action is not None:
            action, move = self.sample_to_action(sample=action)
            action_result = self.take_action(action=action, move=move)
        else:
            action_result = self.take_action(action=None, move=None)

        if not action_result:
            reward = -1
        else:
            reward = 0
        reward += self.get_reward()

        # New state is the robot sensor data (all variables encountered, unobservable = -1)
        # Do not use first column (time)
        self.state = np.full(12, -1)
        self.state[6] = 0
        self.state[7] = 0
        new_states = self.state_to_numeric(self.robot.sensor_data[-1, 1:])
        for id, new_obs in enumerate(new_states):
            self.state[id] = new_obs

        # set done if ???
        if self.environment.time > self.episode_length:
            episode_over = True
        else:
            episode_over = False

        ob = tuple(self.state.tolist())
        info = {}
        print("reward: ", reward)

        return ob, reward, episode_over, info

    def reset(self,
              *,
              seed=None,
              return_info=False,
              options=None,
              ):

        # reset state (switch states, robot position)
        self.configure_environment()
        info = {}

        if return_info:
            return self.initial_state, info
        return self.initial_state

    def state_to_numeric(self, state):
        numeric_state = []
        for x in state:
            # try:
            if type(x) == int:
                numeric_state.append(x)
            elif "waypoint" in x:
                numeric_state.append(int(x[9:]))
            elif "room" in x:
                numeric_state.append(ord(x[-1]) - 97)
            elif len(x) <= 2:
                numeric_state.append(int(x))
        return np.array(numeric_state)

    def render(self):
        if self.ax_env is None:
            fig = plt.figure(1, figsize=(20, 10))
            self.ax_env = fig.add_subplot(1, 1, 1)
        self.environment.plot_environment(ax_env=self.ax_env)

    def get_reward(self):
        reward = 0
        # Search changed values returns variable_name, value pairs
        changed_variables_and_values = self.robot.reasoner.search_changed_values()
        if changed_variables_and_values is not None:
            for row in changed_variables_and_values:
                # For each light that changed value apply a reward or punshment
                if "light" in row[0]:
                    # +1 for light on -1 for light off
                    reward += int(row[1]) * 2 - 1
        return reward

    # Choose initial state of the environment and pass it for constructor
    def configure_environment(self):
        # Clone environment from classes.py
        self.environment = environment_2_rooms.environment_1
        self.robot = environment_2_rooms.robot_1
        self.reasoner = environment_2_rooms.reasoner_1
        self.robot.environment = self.environment
        self.robot.reasoner = self.reasoner

        # robot position
        room = self.robot.room

        # set action space
        actions = self.environment.pass_intervenable_variables()
        moves = self.environment.waypoint_list
        self.action_space = self.convert_actions_to_action_space(actions=actions[:-1], moves=moves)
        self.action_names = actions.tolist()

        # observation_space is observable values
        vars = self.environment.variable_list[:-2]
        self.observation_space = gym.spaces.Tuple([gym.spaces.Discrete(3, start=-1) for _ in vars[:5]] +
                                                  [gym.spaces.Discrete(len(self.environment.waypoint_list))] +
                                                  [gym.spaces.Discrete(len(self.environment.room_list))] +
                                                  [gym.spaces.Discrete(3, start=-1) for _ in vars[5:]])
        self.state = self.environment.get_values(self.environment.variable_list[:5]) + \
                     [self.robot.position.name] + \
                     [self.robot.room.name] + \
                     self.environment.get_values(self.environment.variable_list[5:-2])
        self.state = self.state_to_numeric(self.state)

    # Convert the sampled action space into a action, value pair that the environment can use
    def sample_to_action(self, sample):
        filter = np.nonzero(sample)[0]
        action_name = np.array(self.action_names)[filter]
        if len(action_name) > 1:
            action_name = random.choice(action_name)
        if len(action_name) == 0:
            return None, None

        # If the action is moving, then extract the non-zero value from the list
        # and find the correct waypoint that corresponds to it
        if "position" in action_name[0]:
            wp_list = [wp.name for wp in self.environment.waypoint_list]
            nth_waypoint = np.array(sample)[np.nonzero(sample)[0]]
            action_value = np.array(wp_list)[nth_waypoint]
            return None, [action_name[0], action_value[0]]
        else:
            possible_actions = self.environment.possible_actions(self.robot.position.name)
            action = possible_actions[possible_actions[:, 0] == action_name, :]
            if action is not None:
                if len(action) == 0:
                    return None, None
                action = action[0]
            return action, None

    def take_action(self, action, move=None):
        print("action: ", action, " move: ", move)
        if move is not None:
            move = move[1]
        self.environment.do_actions_move_time(action=action, move=move)

        # Returns whether an action has been performed
        # Used to give the robot a reward based on correct action picking
        if action is None and move is None:
            return False
        return True

    # Takes a set of actions and moves and translates to an action space
    # TODO: apply masking of invalid actions or negative reward for impossible moves
    def convert_actions_to_action_space(self, actions, moves):
        # Either flip a switch or move to a location
        return gym.spaces.Tuple([gym.spaces.Discrete(2) for _ in actions.tolist()] + \
                                [gym.spaces.Discrete(len(moves))])

    # Returns a boolean array that has the indexes for valid actions
    # Can be used to mask the action space
    def pass_valid_actions(self):
        possible_actions = self.environment.possible_actions(self.robot.position)
        possible_moves = self.environment.possible_moves()
        return [action in possible_actions + possible_moves for action in self.action_names]
