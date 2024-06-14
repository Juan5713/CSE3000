from four_room.shortest_path import find_all_action_values
from four_room.utils import obs_to_state
from four_room.wrappers import gym_wrapper
from four_room.env import FourRoomsEnv
from pyvirtualdisplay import Display
from typing import Dict
import imageio
import dill
import gymnasium as gym
import numpy as np
import os

# register the environment and open the training configurations
gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

with open('../../../four_room/configs/fourrooms_train_config.pl', 'rb') as file:
    train_config = dill.load(file)

# create the environment
wrapped_env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1',
                                   agent_pos=train_config['agent positions'],
                                   goal_pos=train_config['goal positions'],
                                   doors_pos=train_config['topologies'],
                                   agent_dir=train_config['agent directions'],
                                   render_mode="rgb_array"))

with open('../../../four_room/configs/fourrooms_test_0_config.pl', 'rb') as file:
    test_config = dill.load(file)

# create the environment
test_env = gym_wrapper((gym.make('MiniGrid-FourRooms-v1',
                                 agent_pos=test_config['agent positions'],
                                 goal_pos=test_config['goal positions'],
                                 doors_pos=test_config['topologies'],
                                 agent_dir=test_config['agent directions'],
                                 render_mode="rgb_array")))

# dataset in d4rl format, that is to say dictionary of numpy arrays
data: Dict[str, list] = {
    "observations": [],
    "actions": [],
    "rewards": [],
    "terminals": [],
    "next_observations": []
}

test_data: Dict[str, list] = {
    "observations": [],
    "actions": [],
    "rewards": [],
    "terminals": [],
    "next_observations": []
}


def fill_dict(to_fill: Dict[str, list], env: FourRoomsEnv, config):
    for i in range(len(config['topologies'])):
        done = False
        # reset the environment once (random list index) and generate the grid for the environment (happens in
        # super.reset)
        obs, _ = env.reset()
        # loop until we reach a goal position
        while not done:
            # images.append(img)
            state = obs_to_state(obs)
            # add the current state before the observation changes
            to_fill["observations"].append(obs.flatten())

            q_values = find_all_action_values(state[:2], state[2], state[3:5], state[5:], 0.99)
            optimal_action = np.argmax(q_values)  # left = 0, right = 1, forward = 2
            obs, reward, terminated, truncated, info = env.step(optimal_action)
            done = terminated or truncated
            # img = wrapped_env.render()

            # so we need to add to the dataset the state, action, reward, whether it reaches terminal and next state
            to_fill["actions"].append(optimal_action)
            to_fill["rewards"].append(reward)
            to_fill["terminals"].append(terminated)
            to_fill["next_observations"].append(obs.flatten())

    data_np: Dict[str, np.ndarray] = {}
    for key in to_fill:
        data_np[key] = np.array(to_fill[key])
    return data_np


# test
# images = []
# img = wrapped_env.render()

# with Display(visible=False) as disp:
#     imageio.mimsave('rendered_episode.gif', [np.array(img) for i, img in enumerate(images) if i % 1 == 0], duration=200)

def gen_dataset():
    return fill_dict(data, wrapped_env, train_config), fill_dict(test_data, test_env, test_config)
