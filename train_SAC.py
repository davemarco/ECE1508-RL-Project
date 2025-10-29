# @title Import packages for plotting and creating graphics
import itertools
import time
from typing import Callable, List, NamedTuple, Optional, Union
import numpy as np

# Graphics and plotting.
import mediapy as media
import matplotlib.pyplot as plt

from datetime import datetime
import functools
import os
from tqdm import tqdm
from IPython.display import display
from typing import Any, Dict, Sequence, Tuple, Union
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.io import html, mjcf, model
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from etils import epath
from flax import struct
from flax.training import orbax_utils
from IPython.display import HTML, clear_output
import jax
from jax import numpy as jp
from matplotlib import pyplot as plt
import mediapy as media
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
from orbax import checkpoint as ocp

from mujoco_playground.config import dm_control_suite_params
from mujoco_playground import registry
from mujoco_playground import wrapper


def progress(num_steps, metrics, times, x_data, y_data, y_dataerr):
    clear_output(wait=True)

    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    y_dataerr.append(metrics["eval/episode_reward_std"])

    plt.xlim([0, sac_params["num_timesteps"] * 1.25])
    plt.ylim([0, 1100])
    plt.xlabel("# environment steps")
    plt.ylabel("reward per episode")
    plt.title(f"y={y_data[-1]:.3f}")
    plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")

    # display(plt.gcf())
    plt.savefig(f"sac_progress_{num_steps}.png")
    plt.close()

def train_sac(env, sac_params, times):
    x_data, y_data, y_dataerr = [], [], []
    sac_training_params = dict(sac_params)
    network_factory = sac_networks.make_sac_networks
    if "network_factory" in sac_params:
        del sac_training_params["network_factory"]
        network_factory = functools.partial(
            sac_networks.make_sac_networks,
            **sac_params.network_factory
        )
    
    train_fn = functools.partial(
        sac.train, **dict(sac_training_params),
        network_factory=network_factory,
        progress_fn=functools.partial(progress, times=times, x_data=x_data, y_data=y_data, y_dataerr=y_dataerr)
    )

    print("Training SAC...")
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")

    rng = jax.random.PRNGKey(0)
    rollout = []
    n_episodes = 1
    
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

    for _ in tqdm(range(n_episodes), total=n_episodes, desc="Collecting rollout"):
        state = jit_reset(rng)
        rollout.append(state)
        for i in range(env_cfg.episode_length):
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, ctrl)
            rollout.append(state)
    
    render_every = 1
    frames = env.render(rollout[::render_every])
    rewards = [s.reward for s in rollout]
    media.show_video(frames, fps=1.0 / env.dt / render_every)
    print(f"mean reward: {np.mean(rewards)}")
    return


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    
    env_name = 'HumanoidWalk'

    env_cfg = registry.get_default_config(env_name)
    env = registry.load(env_name, config=env_cfg)

    sac_params = dm_control_suite_params.brax_sac_config(env_name)
    
    times = [datetime.now()]
    train_sac(env, sac_params, times)