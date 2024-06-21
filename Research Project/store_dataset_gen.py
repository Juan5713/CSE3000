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
import pickle
import os
import random
import math
from d3rlpy.dataset import MDPDataset
from stable_baselines3 import DQN

# register the environment and open the training configurations
gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

print(os.getcwd())

with open('four_room/configs/fourrooms_train_config.pl', 'rb') as file:
    train_config = dill.load(file)

# create the environment
wrapped_env = gym_wrapper(gym.make('MiniGrid-FourRooms-v1',
                                   agent_pos=train_config['agent positions'],
                                   goal_pos=train_config['goal positions'],
                                   doors_pos=train_config['topologies'],
                                   agent_dir=train_config['agent directions'],
                                   render_mode="rgb_array"))

# dataset in d4rl format, that is to say dictionary of numpy arrays
data: Dict[str, list] = {
    "observations": [],
    "actions": [],
    "rewards": [],
    "terminals": [],
    "next_observations": []
}

suboptimal_model = DQN.load("models/dqn_four_rooms_suboptimal")


# -----------------POLICIES-----------------
def expert_policy(state):
    # expert policy: shortest path to goal
    q_values = find_all_action_values(state[:2], state[2], state[3:5], state[5:], 0.99)
    return np.argmax(q_values)


def adversarial_policy(state):
    # the opposite to the expert policy
    q_values = find_all_action_values(state[:2], state[2], state[3:5], state[5:], 0.99)
    return np.argmin(q_values)


def random_policy(env):
    # select a random action
    return env.action_space.sample()


def boltzmann_softmax_policy(state, temperature):
    # boltzmann softmax policy, see http://incompleteideas.net/book/ebook/node17.html
    q_values = find_all_action_values(state[:2], state[2], state[3:5], state[5:], 0.99)
    sum_exp = 0
    for q in q_values:
        sum_exp += math.exp(q / temperature)
    probabilities = [math.exp(q / temperature) / sum_exp for q in q_values]
    return np.random.choice([0, 1, 2], p=probabilities)


def single_action_fill(to_fill: Dict[str, list], env: FourRoomsEnv, action, images):
    # step once in the environment with a given action and fill the dataset
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    img = env.render()
    images.append(img)

    # so we need to add to the dataset the state, action, reward, whether it reaches terminal and next state
    to_fill["actions"].append(action)
    to_fill["rewards"].append(reward)
    to_fill["terminals"].append(terminated)
    to_fill["next_observations"].append(obs.flatten())
    return obs, done


def make_np_data(to_fill: Dict[str, list], images):
    # convert the dataset to numpy format and save the episodes as a gif
    with Display(visible=False) as disp:
        imageio.mimsave('rendered_episode.gif', [np.array(img) for i, img in enumerate(images) if i % 1 == 0],
                        duration=200)
    data_np: Dict[str, np.ndarray] = {}
    for key in to_fill:
        data_np[key] = np.array(to_fill[key])
    return data_np


# -----------------DISCLAIMER-----------------
# The functions that rely on some form of stochasticity, such as mixed policy or epsilon greedy, benefit from going
# through the topologies multiple times as they generate different states and thus integrate more diversity in the
# dataset. Diversity helps performance for offline RL algorithms.
# -------------------------------------------

# -----------------FILLING THE DATASET-----------------
def fill_dict_expert(to_fill: Dict[str, list], env: FourRoomsEnv, config, seed):
    """
    Make a dataset according to the expert policy.

    :param to_fill: the dictionary to fill with the dataset
    :param env: the environment to execute our actions in and sample observations from
    :param config: a configuration (set of topologies) of the environment
    :return: the dataset in numpy format
    """
    images = []
    obs, _ = env.reset(seed=seed)
    for i in range(len(config['topologies'])):
        done = False
        # reset the environment once (random list index) and generate the grid for the environment (happens in
        # super.reset)
        images.append(env.render())
        # loop until we reach a goal position
        while not done:
            # images.append(img)
            state = obs_to_state(obs)
            # add the current state before the observation changes
            to_fill["observations"].append(obs.flatten())

            optimal_action = expert_policy(state)
            obs, done = single_action_fill(to_fill, env, optimal_action, images)
        obs, _ = env.reset()

    return make_np_data(to_fill, images)


def fill_dict_mixed(to_fill: Dict[str, list], env: FourRoomsEnv, config, random_action: bool, seed):
    """
    Make a dataset according to a uniformly mixed policy. We select the expert action with 0.5 probability, and we
    select a suboptimal action with probability 0.5. The suboptimal action can either be random or it can be
    from a suboptimal DQN model that we have trained and stopped to reach about 50% of the expert performance.

    :param to_fill: the dictionary to fill with the dataset
    :param env: the environment to execute our actions in and sample observations from
    :param config: a configuration (set of topologies) of the environment
    :param random_action: whether to select a random action or a suboptimal action
    :return: the dataset in numpy format
    """
    images = []
    obs, _ = env.reset(seed=seed)
    for i in range(3 * len(config['topologies'])):
        done = False
        # reset the environment once (random list index) and generate the grid for the environment (happens in
        # super.reset)
        images.append(env.render())
        # loop until we reach a goal position
        while not done:
            # images.append(img)
            state = obs_to_state(obs)
            # add the current state before the observation changes
            to_fill["observations"].append(obs.flatten())
            num = random.randint(0, 1)
            # uniformly mix random and expert actions
            if num == 0:
                action = expert_policy(state)
            else:
                action = random_policy(env) if random_action else suboptimal_model.predict(obs)[0]
            obs, done = single_action_fill(to_fill, env, action, images)
        obs, _ = env.reset()

    return make_np_data(to_fill, images)


def fill_dict_epsilon_greedy(to_fill: Dict[str, list], env: FourRoomsEnv, config, epsilon: float, random_action: bool, seed):
    """
    Make a dataset according to an epsilon greedy policy. We select the expert action with epsilon probability, and we
    select a suboptimal action with probability 1-epsilon. The suboptimal action can either be random or it can be
    from a suboptimal DQN model that we have trained and stopped to reach about 50% of the expert performance.

    :param to_fill: the dictionary to fill with the dataset
    :param env: the environment to execute our actions in and sample observations from
    :param config: a configuration (set of topologies) of the environment
    :param epsilon: the probability of selecting the expert action
    :param random_action: whether to select a random action or a suboptimal action
    :return: the dataset in numpy format
    """
    images = []
    obs, _ = env.reset(seed=seed)
    for i in range(3 * len(config['topologies'])):
        done = False
        # reset the environment once (random list index) and generate the grid for the environment (happens in
        # super.reset)
        images.append(env.render())
        # loop until we reach a goal position
        while not done:
            # images.append(img)
            state = obs_to_state(obs)
            # add the current state before the observation changes
            to_fill["observations"].append(obs.flatten())
            num = random.random()
            # uniformly mix random and expert actions
            if num < epsilon:
                action = expert_policy(state)
            else:
                action = random_policy(env) if random_action else suboptimal_model.predict(obs)[0]
            obs, done = single_action_fill(to_fill, env, action, images)
        obs, _ = env.reset()

    return make_np_data(to_fill, images)


def fill_dict_boltzmann_softmax(to_fill: Dict[str, list], env: FourRoomsEnv, config, temperature, seed):
    """
    Make a dataset according to boltzmann softmax policy. We select actions by their q-values. The higher the q-value
    of an action in comparison to the other actions the higher the probability of picking it. This balances the expert
    policy and the epsilon greedy policy.

    :param to_fill: the dictionary to fill with the dataset
    :param env: the environment to execute our actions in and sample observations from
    :param config: a configuration (set of topologies) of the environment
    :param temperature: temperature for Boltzmann distribution (the higher the temperature the more random the actions)
    :return: the dataset in numpy format
    """
    images = []
    obs, _ = env.reset(seed=seed)
    for i in range(3 * len(config['topologies'])):
        done = False
        # reset the environment once (random list index) and generate the grid for the environment (happens in
        # super.reset)
        images.append(env.render())
        # loop until we reach a goal position
        while not done:
            # images.append(img)
            state = obs_to_state(obs)
            # add the current state before the observation changes
            to_fill["observations"].append(obs.flatten())

            optimal_action = boltzmann_softmax_policy(state, temperature)
            obs, done = single_action_fill(to_fill, env, optimal_action, images)
        obs, _ = env.reset()

    return make_np_data(to_fill, images)


def fill_dict_adversarial(to_fill: Dict[str, list], env: FourRoomsEnv, config, epsilon, seed):
    """
    Make a dataset according to an epsilon greedy policy. We select the expert action with epsilon probability, and we
    select an adversarial action with probability 1-epsilon. The adversarial action is the one with the lowest q-value
    i.e. puts us the furthest from the goal.

    :param to_fill: the dictionary to fill with the dataset
    :param env: the environment to execute our actions in and sample observations from
    :param config: a configuration (set of topologies) of the environment
    :param epsilon: the probability of selecting the expert action
    :return: the dataset in numpy format
    """
    images = []
    obs, _ = env.reset(seed=seed)
    for i in range(3 * len(config['topologies'])):
        done = False
        # reset the environment once (random list index) and generate the grid for the environment (happens in
        # super.reset)
        images.append(env.render())
        # loop until we reach a goal position
        while not done:
            # images.append(img)
            state = obs_to_state(obs)
            # add the current state before the observation changes
            to_fill["observations"].append(obs.flatten())
            num = random.random()
            # uniformly mix random and expert actions
            if num < epsilon:
                action = expert_policy(state)
            else:
                action = adversarial_policy(state)
            obs, done = single_action_fill(to_fill, env, action, images)
        obs, _ = env.reset()

    return make_np_data(to_fill, images)


def fill_dict_random(to_fill: Dict[str, list], env: FourRoomsEnv, config, seed):
    """
    Make a dataset according to a fully random policy. Actions are uniformly sampled from the environment action space
    at each step.

    :param to_fill: the dictionary to fill with the dataset
    :param env: the environment to execute our actions in and sample observations from
    :param config: a configuration (set of topologies) of the environment
    :return: the dataset in numpy format
    """
    images = []
    obs, _ = env.reset(seed=seed)
    for i in range(len(config['topologies'])):
        done = False
        # reset the environment once (random list index) and generate the grid for the environment (happens in
        # super.reset)
        images.append(env.render())
        # loop until we reach a goal position
        while not done:
            # add the current state before the observation changes
            to_fill["observations"].append(obs.flatten())

            action = random_policy(env)
            obs, done = single_action_fill(to_fill, env, action, images)
        obs, _ = env.reset()

    return make_np_data(to_fill, images)


# -----------------GENERATING THE DATASET-----------------
def gen_dataset(policy_name, seed, step):
    dataset = None
    if policy_name == 'expert':
        dataset = fill_dict_expert(data, wrapped_env, train_config, seed)
    elif policy_name == 'mixed_random':
        dataset = fill_dict_mixed(data, wrapped_env, train_config, True, seed)
    elif policy_name == 'mixed_suboptimal':
        dataset = fill_dict_mixed(data, wrapped_env, train_config, False, seed)
    elif policy_name == 'egreedy':
        dataset = fill_dict_epsilon_greedy(data, wrapped_env, train_config, 0.75, False, seed)
    elif policy_name == 'boltzmann_05':
        dataset = fill_dict_boltzmann_softmax(data, wrapped_env, train_config, 0.5, seed)
    elif policy_name == 'boltzmann_15':
        dataset = fill_dict_boltzmann_softmax(data, wrapped_env, train_config, 1.5, seed)
    elif policy_name == 'random':
        dataset = fill_dict_random(data, wrapped_env, train_config, seed)
    # create a variant supported by bc
    bc_dataset = MDPDataset(dataset['observations'], dataset['actions'], dataset['rewards'], dataset['terminals'],
                            action_size=3)
    with open('DATASETS2/{}_dataset_iql_{}.pkl'.format(policy_name, step), 'wb') as writeFile:
        # Serialize and save the data to the file
        pickle.dump(dataset, writeFile)
    with open('DATASETS2/{}_dataset_bc_{}.pkl'.format(policy_name, step), 'wb') as writeFile:
        # Serialize and save the data to the file
        pickle.dump(bc_dataset, writeFile)


for i in range(3):
    seeds = [50, 100, 150]
    random.seed(seeds[i])
    gen_dataset('expert', seeds[i], i+1)
    gen_dataset('mixed_random', seeds[i], i+1)
    gen_dataset('mixed_suboptimal', seeds[i], i+1)
    gen_dataset('egreedy', seeds[i], i+1)
    gen_dataset('boltzmann_05', seeds[i], i+1)
    gen_dataset('boltzmann_15', seeds[i], i+1)
    gen_dataset('random', seeds[i], i+1)

