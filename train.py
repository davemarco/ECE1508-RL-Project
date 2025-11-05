from datetime import datetime
import os
import argparse
import json
import functools
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import mediapy as media
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
import jax
import pickle
from ml_collections import config_dict



def progress(num_steps, metrics, pbar, times, x_data, y_data, y_dataerr, output_dir):
    # clear_output(wait=True)
    if pbar is not None:
        pbar.update(1)

    r = metrics.get("eval/episode_reward", None)
    r_std = metrics.get("eval/episode_reward_std", None)
    
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(r)
    y_dataerr.append(r_std)

    plt.xlim([0, params["num_timesteps"] * 1.25])
    plt.ylim([0, 1100])
    plt.xlabel("# environment steps")
    plt.ylabel("reward per episode")
    plt.title(f"y={y_data[-1]:.3f}")
    plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
    plt.savefig(f"{output_dir}/progress_{num_steps}.png")
    plt.close()


def train_ppo(env, ppo_params, times, output_dir):
    x_data, y_data, y_dataerr = [], [], []
    ppo_training_params = dict(ppo_params)
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_params:
        del ppo_training_params["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_params.network_factory
        )
    
    n_updates = int(ppo_params.num_timesteps / (ppo_params.num_envs * ppo_params.unroll_length))
    pbar = tqdm(total=n_updates, desc="Training steps")
    
    train_fn = functools.partial(
        ppo.train, **dict(ppo_training_params),
        network_factory=network_factory,
        progress_fn=functools.partial(progress, 
                                      pbar=pbar, 
                                      params=ppo_params, 
                                      times=times, 
                                      x_data=x_data, 
                                      y_data=y_data, 
                                      y_dataerr=y_dataerr,
                                      output_dir=output_dir)
    )

    print("Training PPO...")
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")

    return make_inference_fn, params, metrics

def train_sac(env, sac_params, times, output_dir):
    x_data, y_data, y_dataerr = [], [], []
    sac_training_params = dict(sac_params)
    network_factory = sac_networks.make_sac_networks
    if "network_factory" in sac_params:
        del sac_training_params["network_factory"]
        network_factory = functools.partial(
            sac_networks.make_sac_networks,
            **sac_params.network_factory
        )
    
    n_updates = int(sac_params.num_timesteps / (sac_params.num_envs * sac_params.unroll_length))
    pbar = tqdm(total=n_updates, desc="Training steps")
    
    train_fn = functools.partial(
        sac.train, **dict(sac_training_params),
        network_factory=network_factory,
        progress_fn=functools.partial(progress, 
                                      pbar=pbar, 
                                      params=sac_params,
                                      times=times, 
                                      x_data=x_data, 
                                      y_data=y_data, 
                                      y_dataerr=y_dataerr,
                                      output_dir=output_dir)
    )

    print("Training SAC...")
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")

    return make_inference_fn, params, metrics


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="HumanoidWalk")
    parser.add_argument("--algo", type=str, default="PPO", choices=["PPO", "SAC"])
    parser.add_argument("--num_timesteps", type=int, default=int(5e8))
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--unroll_length", type=int, default=32, help="Number of steps to unroll the environment for training")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate for the optimizer")
    parser.add_argument("--reward_scaling", type=float, default=1.0, help="Scaling factor for the reward")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save the results")
    parser.add_argument("--action_repeat", type=int, default=1, help="Number of times to repeat the action")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for the training")
    parser.add_argument("--discounting", type=float, default=0.995, help="Discounting factor for the reward")
    parser.add_argument("--entropy_cost", type=float, default=0.01, help="Entropy cost for the training")
    parser.add_argument("--episode_length", type=int, default=1000, help="Length of the episode")
    parser.add_argument("--normalize_observations", type=bool, default=True, help="Normalize the observations")
    parser.add_argument("--num_evals", type=int, default=10, help="Number of evaluations")
    parser.add_argument("--num_minibatches", type=int, default=32, help="Number of minibatches for the training")
    parser.add_argument("--num_updates_per_batch", type=int, default=16, help="Number of updates per batch")
    parser.add_argument("--config_file", type=str, default=None, help="Path to the config file")
    return parser.parse_args()

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    
    args = get_args()
    env_name = args.env

    env_cfg = registry.get_default_config(env_name)
    env = registry.load(env_name, config=env_cfg)
    
    if args.algo == "PPO":
        params = dm_control_suite_params.brax_ppo_config(env_name)
    elif args.algo == "SAC":
        params = dm_control_suite_params.brax_sac_config(env_name)

    #? Load config file if provided
    if args.config_file is not None:
        config = json.load(open(args.config_file))
        config_params = config_dict.ConfigDict(config)
        params.update(config_params)
    
    #? Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    #? Save params to a well-formatted JSON file
    params_dict = params.to_dict()
    with open(f"{args.output_dir}/params_{args.algo}_{env_name}.json", "w") as f:
        json.dump(params_dict, f, indent=4)

    times = [datetime.now()]
    if args.algo == "PPO":
        make_inference_fn, params, metrics = train_ppo(env, params, times, args.output_dir)
    elif args.algo == "SAC":
        make_inference_fn, params, metrics = train_sac(env, params, times, args.output_dir)
    
    #? Save the model
    with open(f"{args.output_dir}/model_{args.algo}_{env_name}.pkl", "wb") as f:
        pickle.dump(make_inference_fn, f)
    
    #? Save the params
    params_dict = params.to_dict()
    with open(f"{args.output_dir}/params_{args.algo}_{env_name}_post_training.json", "w") as f:
        json.dump(params_dict, f, indent=4)
    
    #? Save the metrics
    with open(f"{args.output_dir}/metrics_{args.algo}_{env_name}.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    #? Collect a rollout
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
    out_path = f"rollout_{env_name}.mp4"
    media.write_video(out_path, np.array(frames), fps=1.0 / env.dt / render_every)
    print(f"mean reward: {np.mean(rewards)}")