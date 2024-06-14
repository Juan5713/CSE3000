# source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import copy
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import dill
from torch.distributions import Categorical
from torch.optim.lr_scheduler import CosineAnnealingLR
from four_room.env import FourRoomsEnv
from four_room.wrappers import gym_wrapper
import pickle
import d3rlpy
import seaborn as sns
import argparse
from csv import writer
import optuna
from optuna.visualization import plot_param_importances
import matplotlib.pyplot as plt

TensorBatch = List[torch.Tensor]

EXP_ADV_MAX = 100.0
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


@dataclass
class TrainConfig:
    # Experiment
    device: str = "cuda"
    env: str = "four-room"  # OpenAI gym environment name
    seed: int = 0  # Sets Gym, PyTorch and Numpy seeds
    eval_freq: int = int(25000)  # How often (time steps) we evaluate
    n_episodes: int = 40  # How many episodes run during evaluation
    max_timesteps: int = int(25000)  # Max time steps to run environment
    checkpoints_path: Optional[str] = None  # Save path
    load_model: str = ""  # Model load file name, "" doesn't load
    # IQL
    buffer_size: int = 2_000_000  # Replay buffer size
    batch_size: int = 256  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.006  # Target network update rate, default 0.005
    beta: float = 3.0  # Inverse temperature. Small beta -> BC, big beta -> maximizing Q
    iql_tau: float = 0.8  # Coefficient for asymmetric loss, default 0.7
    iql_deterministic: bool = False  # Use deterministic actor
    normalize: bool = True  # Normalize states
    normalize_reward: bool = False  # Normalize reward
    vf_lr: float = 0.00030631780632380705  # V function learning rate, default 3e-4
    qf_lr: float = 0.0001904542069504683  # Critic learning rate, default 3e-4
    actor_lr: float = 0.004450418567816549  # Actor learning rate, default 3e-4
    actor_dropout: Optional[float] = None  # Adroit uses dropout for policy network
    # Wandb logging
    project: str = "CORL"
    group: str = "IQL-D4RL"
    name: str = "IQL"

    def __post_init__(self):
        self.name = f"{self.name}-{self.env}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def compute_mean_std(states: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray]:
    mean = states.mean(0)
    std = states.std(0) + eps
    return mean, std


def normalize_states(states: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (states - mean) / std


def wrap_env(
        env: gym.Env,
        state_mean: Union[np.ndarray, float] = 0.0,
        state_std: Union[np.ndarray, float] = 1.0,
        reward_scale: float = 1.0,
) -> gym.Env:
    # PEP 8: E731 do not assign a lambda expression, use a def
    def normalize_state(state):
        return (
                state.flatten() - state_mean
        ) / state_std  # epsilon should be already added in std.

    def scale_reward(reward):
        # Please be careful, here reward is multiplied by scale!
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


class ReplayBuffer:
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            buffer_size: int,
            device: str = "cpu",
    ):
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0

        self._states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            (buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            (buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"][..., None])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._pointer = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._pointer), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError


def set_seed(
        seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        # env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )
    wandb.run.save()


@torch.no_grad()
def eval_actor(
        env: gym.Env, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    # env.seed(seed)
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, _ = env.reset(seed=seed)
        done = False
        episode_reward = 0.0
        while not done:
            action = actor.act(state, device)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


def modify_reward(dataset, env_name, max_episode_steps=1000):
    if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
        min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
        dataset["rewards"] /= max_ret - min_ret
        dataset["rewards"] *= max_episode_steps
    elif "antmaze" in env_name:
        dataset["rewards"] -= 1.0


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class MLP(nn.Module):
    def __init__(
            self,
            dims,
            activation_fn: Callable[[], nn.Module] = nn.ReLU,
            output_activation_fn: Callable[[], nn.Module] = None,
            squeeze_output: bool = False,
            dropout: Optional[float] = None,
    ):
        super().__init__()
        n_dims = len(dims)
        if n_dims < 2:
            raise ValueError("MLP requires at least two dims (input and output)")

        layers = []
        for i in range(n_dims - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())

            if dropout is not None:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            if dims[-1] != 1:
                raise ValueError("Last dim must be 1 when squeezing")
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(
            self,
            state_dim: int,
            act_dim: int,
            max_action: float,
            hidden_dim: int = 256,
            n_hidden: int = 2,
            dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32))
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> Categorical:
        mean_logit = self.net(obs)
        # std = torch.exp(self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX))
        return Categorical(logits=mean_logit)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        dist = self(state)
        action = torch.argmax(dist.probs[0]) if not self.training else dist.sample()
        # action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()


class DeterministicPolicy(nn.Module):
    def __init__(
            self,
            state_dim: int,
            act_dim: int,
            max_action: float,
            hidden_dim: int = 256,
            n_hidden: int = 2,
            dropout: Optional[float] = None,
    ):
        super().__init__()
        self.net = MLP(
            [state_dim, *([hidden_dim] * n_hidden), act_dim],
            output_activation_fn=nn.Tanh,
            dropout=dropout,
        )
        self.max_action = max_action

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str = "cpu"):
        state = torch.tensor(state.reshape(1, -1), device=device, dtype=torch.float32)
        return (
            torch.clamp(self(state) * self.max_action, -self.max_action, self.max_action)
            .cpu()
            .data.numpy()
            .flatten()
        )


class TwinQ(nn.Module):
    def __init__(
            self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden: int = 2
    ):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), action_dim]
        self.q1 = MLP(dims)
        self.q2 = MLP(dims)

    def both(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q1(state), self.q2(state)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.min(*self.both(state))


class ValueFunction(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [state_dim, *([hidden_dim] * n_hidden), 1]
        self.v = MLP(dims, squeeze_output=True)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state)


class ImplicitQLearning:
    def __init__(
            self,
            max_action: float,
            actor: nn.Module,
            actor_optimizer: torch.optim.Optimizer,
            q_network: nn.Module,
            q_optimizer: torch.optim.Optimizer,
            v_network: nn.Module,
            v_optimizer: torch.optim.Optimizer,
            iql_tau: float = 0.7,
            beta: float = 3.0,
            max_steps: int = 1000000,
            discount: float = 0.99,
            tau: float = 0.005,
            device: str = "cpu",
    ):
        self.max_action = max_action
        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)
        self.iql_tau = iql_tau
        self.beta = beta
        self.discount = discount
        self.tau = tau

        self.total_it = 0
        self.device = device

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = torch.gather(self.q_target(observations), 1, actions.long())

        v = self.vf(observations)
        adv = target_q[:, 0] - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
            self,
            next_v: torch.Tensor,
            observations: torch.Tensor,
            actions: torch.Tensor,
            rewards: torch.Tensor,
            terminals: torch.Tensor,
            log_dict: Dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations)
        q_loss = sum(F.mse_loss(q[:, 0], targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
            self,
            adv: torch.Tensor,
            observations: torch.Tensor,
            actions: torch.Tensor,
            log_dict: Dict,
    ):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations)
        if isinstance(policy_out, torch.distributions.Distribution):
            bc_losses = -policy_out.log_prob(actions[:, 0]).sum(-1, keepdim=False)
        elif torch.is_tensor(policy_out):
            if policy_out.shape != actions.shape:
                raise RuntimeError("Actions shape missmatch")
            bc_losses = torch.sum((policy_out - actions) ** 2, dim=1)
        else:
            raise NotImplementedError
        policy_loss = torch.mean(exp_adv * bc_losses)
        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            rewards,
            next_observations,
            dones,
        ) = batch
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        # Update value function
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        # Update actor
        self._update_policy(adv, observations, actions, log_dict)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"]


def train_bc(dataset, seed, steps, learning_rate=0.000701):
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args()

    d3rlpy.seed(seed)

    bc = d3rlpy.algos.DiscreteBCConfig(learning_rate=learning_rate
                                       ).create(device=args.gpu)

    bc.fit(
        dataset,
        n_steps=steps,
        n_steps_per_epoch=1000
    )

    return bc


def eval_bc(env, test_config, bc, seed):
    rewards = []
    for i in range(len(test_config['topologies'])):
        obs, _ = env.reset(seed=seed)
        done = False
        steps = 0
        while not done:
            steps += 1
            obs_flattened = obs.flatten()[None, :]
            action = bc.predict(obs_flattened)
            obs, reward, terminated, truncated, info = env.step(action[0])
            done = terminated or truncated
            if done:
                rewards.append(reward)
    return rewards


@pyrallis.wrap()
def train(config: TrainConfig, iql_dataset_path, bc_dataset_path, policy_name, seed,
          steps, learning_rates=None, taus=None, tuning=False):
    # only passed when tuning hyperparameters
    if taus is not None:
        config.tau = taus[0]
        config.beta = taus[1]
        config.iql_tau = taus[2]
    if learning_rates is not None:
        config.vf_lr = learning_rates[0]
        config.qf_lr = learning_rates[1]
        config.actor_lr = learning_rates[2]

    config.seed = seed
    config.max_timesteps = steps
    config.eval_freq = steps

    # register the environment and open the training configurations
    gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

    with open('../../../four_room/configs/fourrooms_train_config.pl', 'rb') as file:
        train_config = dill.load(file)

    # create the environment
    env = gym_wrapper((gym.make('MiniGrid-FourRooms-v1',
                                agent_pos=train_config['agent positions'],
                                goal_pos=train_config['goal positions'],
                                doors_pos=train_config['topologies'],
                                agent_dir=train_config['agent directions'],
                                render_mode="rgb_array")))

    # make the testing environment
    with open('../../../four_room/configs/fourrooms_test_0_config.pl', 'rb') as file:
        test_config = dill.load(file)

    # create the environment
    test_env = gym_wrapper((gym.make('MiniGrid-FourRooms-v1',
                                     agent_pos=test_config['agent positions'],
                                     goal_pos=test_config['goal positions'],
                                     doors_pos=test_config['topologies'],
                                     agent_dir=test_config['agent directions'],
                                     render_mode="rgb_array")))

    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
    action_dim = env.action_space.n

    with open(iql_dataset_path, 'rb') as readFile:
        # Serialize and save the data to the file
        dataset = pickle.load(readFile)

    with open(bc_dataset_path, 'rb') as readFile:
        # Serialize and save the data to the file
        bc_dataset = pickle.load(readFile)

    if config.normalize_reward:
        modify_reward(dataset, env)

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )

    env = wrap_env(env, state_mean=state_mean, state_std=state_std)

    replay_buffer = ReplayBuffer(
        state_dim,
        action_dim,
        config.buffer_size,
        config.device,
    )
    replay_buffer.load_d4rl_dataset(dataset)

    max_action = float(2)

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    set_seed(seed, env)

    q_network = TwinQ(state_dim, action_dim).to(config.device)
    v_network = ValueFunction(state_dim).to(config.device)
    actor = (
        DeterministicPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
        if config.iql_deterministic
        else GaussianPolicy(
            state_dim, action_dim, max_action, dropout=config.actor_dropout
        )
    ).to(config.device)
    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr)

    kwargs = {
        "max_action": max_action,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "q_network": q_network,
        "q_optimizer": q_optimizer,
        "v_network": v_network,
        "v_optimizer": v_optimizer,
        "discount": config.discount,
        "tau": config.tau,
        "device": config.device,
        # IQL
        "beta": config.beta,
        "iql_tau": config.iql_tau,
        "max_steps": config.max_timesteps,
    }

    print("---------------------------------------")
    print(f"Training IQL, Env: {config.env}, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = ImplicitQLearning(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    # wandb_init(asdict(config))

    # initial train of the bc
    bc = train_bc(bc_dataset, seed, 100)
    curr_bc_steps = 100
    # do all eval steps in one go
    step_array = [100, 200, 500]

    # evaluations = []
    for t in range(int(config.max_timesteps)):
        batch = replay_buffer.sample(config.batch_size)
        batch = [b.to(config.device) for b in batch]
        log_dict = trainer.train(batch)
        # wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.eval_freq == 0:
            # print(f"Time steps: {t + 1}")
            # re-train bc after each eval run
            if t + 1 != 100:
                bc.fit(
                    bc_dataset,
                    n_steps=t + 1 - curr_bc_steps,
                    n_steps_per_epoch=100
                )
                curr_bc_steps = t + 1
            # EVALUATE BOTH IQL AND BC, DO IT 10 TIMES TO GET THE AVERAGE
            iql_scores = []
            bc_scores = []
            for i in range(10):
                eval_scores = eval_actor(
                    test_env,
                    actor,
                    device=config.device,
                    n_episodes=config.n_episodes,
                    seed=seed,
                )
                iql_scores.append(sum(eval_scores))
                bc_scores.append(sum(eval_bc(test_env, test_config, bc, seed)))
            # normalized_eval_score = env.get_normalized_score(eval_score) * 100.0
            # evaluations.append(eval_score)
            # print("---------------------------------------")
            # print(
            #     f"Evaluation over {config.n_episodes} episodes: "
            #     f"{eval_score:.3f} , D4RL score: {eval_score:.3f}"
            # )
            # print("---------------------------------------")
            if config.checkpoints_path is not None:
                torch.save(
                    trainer.state_dict(),
                    os.path.join(config.checkpoints_path, f"checkpoint_{t}.pt"),
                )

            if not tuning:
                file_name = "../../../RESULTS/tuned_unreachable/unreachable_full_{}.csv".format(t + 1)
                row_contents = [sum(iql_scores) / 10, sum(bc_scores) / 10, policy_name,
                                dataset["observations"].shape[0], t + 1, seed]
                with open(file_name, 'a+', newline='') as write_obj:
                    csv_writer = writer(write_obj)
                    csv_writer.writerow(row_contents)
                # wandb.log(
                #     {"d4rl_normalized_score": eval_score}, step=trainer.total_it
                # )
            else:
                return sum(iql_scores) / 10


# hyperparameter tuning
def objective(trial):
    # learning rates
    vf_lr: float = trial.suggest_loguniform('vf_lr', 1e-4, 1e-2)
    qf_lr: float = trial.suggest_loguniform('qf_lr', 1e-4, 1e-2)
    actor_lr: float = trial.suggest_loguniform('actor_lr', 1e-4, 1e-2)
    # other iql parameters
    tau: float = trial.suggest_float('tau', 0.003, 0.007, step=0.001)  # Target network update rate
    beta: float = trial.suggest_float('beta', 1.0, 3.0, step=0.5)  # Inverse temperature
    iql_tau: float = trial.suggest_float('iql_tau', 0.5, 1.0, step=0.1)  # Coefficient for asymmetric loss

    learning_rates = [vf_lr, qf_lr, actor_lr]
    taus = [tau, beta, iql_tau]

    trial_seed: int = trial.suggest_int('trial_seed', 5, 10, step=1)

    max_steps: int = trial.suggest_int('max_steps', 1000, 25000, step=1000)

    return train(iql_dataset_path='../../../DATASETS/mixed_dataset_suboptimal_iql_1.pkl',
                 bc_dataset_path='../../../DATASETS/mixed_dataset_suboptimal_bc_1.pkl',
                 policy_name='Mixed Suboptimal', seed=trial_seed, steps=max_steps, learning_rates=learning_rates,
                 taus=taus, tuning=True)


def objective_bc(trial, bc_dataset_path):
    # learning rate, default is 1e-3
    learning_rate: float = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    # amount of steps
    max_steps: int = trial.suggest_int('max_steps', 1000, 25000, step=1000)
    # seed
    trial_seed: int = trial.suggest_int('trial_seed', 5, 10, step=1)

    with open(bc_dataset_path, 'rb') as readFile:
        # Serialize and save the data to the file
        bc_dataset = pickle.load(readFile)
    bc = train_bc(bc_dataset, trial_seed, max_steps, learning_rate)
    # make the testing environment
    gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)
    with open('../../../four_room/configs/fourrooms_test_0_config.pl', 'rb') as file:
        test_config = dill.load(file)

    # create the environment
    test_env = gym_wrapper((gym.make('MiniGrid-FourRooms-v1',
                                     agent_pos=test_config['agent positions'],
                                     goal_pos=test_config['goal positions'],
                                     doors_pos=test_config['topologies'],
                                     agent_dir=test_config['agent directions'],
                                     render_mode="rgb_array")))

    bc_scores = []
    for j in range(10):
        bc_scores.append(sum(eval_bc(test_env, test_config, bc, trial_seed)))
    return sum(bc_scores) / 10


if __name__ == "__main__":
    tuning = True
    if not tuning:
        # 0 and 4 random prime numbers
        seeds = [0, 4219, 17333, 39779, 44987]
        curr_steps = 500
        for k in range(5):
            seed = seeds[k]
            # train the expert dataset
            train(iql_dataset_path='../../../DATASETS/expert_dataset_iql.pkl',
                  bc_dataset_path='../../../DATASETS/expert_dataset_bc.pkl',
                  policy_name='Expert', seed=seed, steps=curr_steps, tuning=tuning)
            # train all the other datasets, two datasets each policy to account for a bit of stochasticity
            for i in range(2):
                train(iql_dataset_path='../../../DATASETS/mixed_dataset_suboptimal_iql_{}.pkl'.format(i + 1),
                      bc_dataset_path='../../../DATASETS/mixed_dataset_suboptimal_bc_{}.pkl'.format(i + 1),
                      policy_name='Mixed Suboptimal', seed=seed, steps=curr_steps, tuning=tuning)
                train(iql_dataset_path='../../../DATASETS/mixed_dataset_random_iql_{}.pkl'.format(i + 1),
                      bc_dataset_path='../../../DATASETS/mixed_dataset_random_bc_{}.pkl'.format(i + 1),
                      policy_name='Mixed Random', seed=seed, steps=curr_steps, tuning=tuning)
                train(iql_dataset_path='../../../DATASETS/egreedy_dataset_suboptimal_iql_{}.pkl'.format(i + 1),
                      bc_dataset_path='../../../DATASETS/egreedy_dataset_suboptimal_bc_{}.pkl'.format(i + 1),
                      policy_name='Epsilon-greedy', seed=seed, steps=curr_steps, tuning=tuning)
                train(iql_dataset_path='../../../DATASETS/boltzmann_dataset_05_iql_{}.pkl'.format(i + 1),
                      bc_dataset_path='../../../DATASETS/boltzmann_dataset_05_bc_{}.pkl'.format(i + 1),
                      policy_name='Boltzmann Softmax: Tau 0.5', seed=seed, steps=curr_steps, tuning=tuning)
                train(iql_dataset_path='../../../DATASETS/boltzmann_dataset_15_iql_{}.pkl'.format(i + 1),
                      bc_dataset_path='../../../DATASETS/boltzmann_dataset_15_bc_{}.pkl'.format(i + 1),
                      policy_name='Boltzmann Softmax: Tau 1.5', seed=seed, steps=curr_steps, tuning=tuning)
                train(iql_dataset_path='../../../DATASETS/random_dataset_iql_{}.pkl'.format(i + 1),
                      bc_dataset_path='../../../DATASETS/random_dataset_bc_{}.pkl'.format(i + 1),
                      policy_name='Random', seed=seed, steps=curr_steps, tuning=tuning)
    else:
        # hyperparameter tuning
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial),
                       n_trials=150)
        print("---------------------------------------")
        print("Best params are: ")
        print(study.best_params)
        print("Best value is: ")
        # current best IQL subopt: 'vf_lr': 0.00030631780632380705, 'qf_lr': 0.0001904542069504683, 'actor_lr':
        # 0.004450418567816549, 'tau': 0.006, 'beta': 3.0, 'iql_tau': 0.8
        # current best for BC: 'learning_rate': 0.000701
        print(study.best_value)

        fig = plot_param_importances(study)
        fig.update_layout(title="Hyperparameter Importances for IQL")
        fig.write_image("hyperparameter_importances_iql.png")