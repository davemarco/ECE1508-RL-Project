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

def progress(num_steps, metrics, pbar, times, x_data, y_data, y_dataerr, output_dir, total_timesteps):
    # Only log/plot on process 0 if multi-host (optional).
    if pbar is not None:
        delta = num_steps - pbar.n
        pbar.update(delta)

    r = metrics.get("eval/episode_reward", None)
    r_std = metrics.get("eval/episode_reward_std", None)
    
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(r)
    y_dataerr.append(r_std)

    plt.xlim([0, total_timesteps * 1.25])
    plt.ylim([0, 1100])
    plt.xlabel("# environment steps")
    plt.ylabel("reward per episode")
    plt.title(f"y={y_data[-1]:.3f}")
    plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")
    plt.savefig(f"{output_dir}/progress.png")
    plt.close()

def make_env(env_name, episode_length, action_repeat):
    env_cfg = registry.get_default_config(env_name)
    env_cfg.episode_length = episode_length
    env_cfg.action_repeat = action_repeat
    env = registry.load(env_name, config=env_cfg)
    return env

# Wrapper that matches Brax's expected signature; accepts env instance or factory.
def my_env_wrap(env, **kwargs):
    # If a factory (e.g., functools.partial) is provided, instantiate it.
    if not hasattr(env, "reset") or not hasattr(env, "step"):
        env = env()
    return wrapper.wrap_for_brax_training(env, **kwargs)

def train_ppo(env_maker, ppo_cfg, times, output_dir):
    x_data, y_data, y_dataerr = [], [], []
    # Keep config/learned params distinct.
    ppo_training_cfg = dict(ppo_cfg)
    network_factory = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_cfg:
        del ppo_training_cfg["network_factory"]
        network_factory = functools.partial(
            ppo_networks.make_ppo_networks,
            **ppo_cfg.network_factory
        )
    n_updates = int(ppo_cfg.num_timesteps / (ppo_cfg.num_envs * ppo_cfg.unroll_length))
    pbar = tqdm(total=n_updates, desc="Training steps")
    total_timesteps = ppo_cfg.num_timesteps

    train_fn = functools.partial(
        ppo.train,
        **ppo_training_cfg,
        network_factory=network_factory,
        progress_fn=functools.partial(progress, 
                                      pbar=pbar, 
                                      times=times, 
                                      x_data=x_data, 
                                      y_data=y_data, 
                                      y_dataerr=y_dataerr,
                                      output_dir=output_dir,
                                      total_timesteps=total_timesteps)
    )
    print("Training PPO...")
    make_inference_fn, train_params, metrics = train_fn(
        environment=env_maker,
        wrap_env_fn=my_env_wrap
    )
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")
    return make_inference_fn, train_params, metrics, x_data, y_data, y_dataerr

def train_sac(env_maker, sac_cfg, times, output_dir):
    x_data, y_data, y_dataerr = [], [], []
    sac_training_cfg = dict(sac_cfg)
    network_factory = sac_networks.make_sac_networks
    if "network_factory" in sac_cfg:
        del sac_training_cfg["network_factory"]
        network_factory = functools.partial(
            sac_networks.make_sac_networks,
            **sac_cfg.network_factory
        )
    n_updates = int(sac_cfg.num_timesteps / (sac_cfg.num_envs * sac_cfg.grad_updates_per_step))
    pbar = tqdm(total=n_updates, desc="Training steps")
    total_timesteps = sac_cfg.num_timesteps

    train_fn = functools.partial(
        sac.train, 
        **sac_training_cfg,
        network_factory=network_factory,
        progress_fn=functools.partial(progress, 
                                      pbar=pbar, 
                                      times=times, 
                                      x_data=x_data, 
                                      y_data=y_data, 
                                      y_dataerr=y_dataerr,
                                      output_dir=output_dir,
                                      total_timesteps=total_timesteps)
    )
    print("Training SAC...")
    make_inference_fn, train_params, metrics = train_fn(
        environment=env_maker,
        wrap_env_fn=my_env_wrap
    )
    print(f"time to jit: {times[1] - times[0]}")
    print(f"time to train: {times[-1] - times[1]}")
    return make_inference_fn, train_params, metrics, x_data, y_data, y_dataerr

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
    # parser.add_argument("--seed", type=int, default=0, help="Seed for the training")
    return parser.parse_args()

if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=100)
    args = get_args()
    env_name = args.env

    # On multi-device: ensure num_envs divisible by jax.local_device_count()
    device_count = jax.local_device_count()
    assert args.num_envs % device_count == 0, f"num_envs ({args.num_envs}) not divisible by device count ({device_count})"

    # Env maker for device-safe replication.
    env_maker = functools.partial(make_env, env_name, args.episode_length, args.action_repeat)

    # Configuration object
    if args.algo == "PPO":
        ppo_cfg = dm_control_suite_params.brax_ppo_config(env_name)
        # Override from CLI/config
        # ppo_cfg.update(vars(args))
        if args.config_file is not None:
            config = json.load(open(args.config_file))
            config_params = config_dict.ConfigDict(config)
            ppo_cfg.update(config_params)
        params = ppo_cfg
    elif args.algo == "SAC":
        sac_cfg = dm_control_suite_params.brax_sac_config(env_name)
        # Override from CLI/config
        # sac_cfg.update(vars(args))
        if args.config_file is not None:
            config = json.load(open(args.config_file))
            config_params = config_dict.ConfigDict(config)
            sac_cfg.update(config_params)
        params = sac_cfg

    os.makedirs(args.output_dir, exist_ok=True)
    params_dict = params.to_dict()
    with open(f"{args.output_dir}/params_{args.algo}_{env_name}.json", "w") as f:
        json.dump(params_dict, f, indent=4)

    times = [datetime.now()]
    if args.algo == "PPO":
        make_inference_fn, trained_params, metrics, x_data, y_data, y_dataerr = train_ppo(env_maker, params, times, args.output_dir)
    elif args.algo == "SAC":
        make_inference_fn, trained_params, metrics, x_data, y_data, y_dataerr = train_sac(env_maker, params, times, args.output_dir)
        

    # #? Save the model
    # with open(f"{args.output_dir}/model_{args.algo}_{env_name}.pkl", "wb") as f:
    #     pickle.dump(make_inference_fn, f)

    # #? Save the params
    # if isinstance(params, tuple):
    #     params = params[0]
    # params_dict = params.to_dict()
    # with open(f"{args.output_dir}/params_{args.algo}_{env_name}_post_training.json", "w") as f:
    #     json.dump(params_dict, f, indent=4)

    # #? Save the metrics
    # with open(f"{args.output_dir}/metrics_{args.algo}_{env_name}.json", "w") as f:
    #     json.dump(metrics, f, indent=4)

    

    #? Save the x_data, y_data, y_dataerr

    with open(f"{args.output_dir}/x_data_{args.algo}_{env_name}.npy", "wb") as f:
        np.save(f, np.array(x_data))
    with open(f"{args.output_dir}/y_data_{args.algo}_{env_name}.npy", "wb") as f:
        np.save(f, np.array(y_data))
    with open(f"{args.output_dir}/y_dataerr_{args.algo}_{env_name}.npy", "wb") as f:
        np.save(f, np.array(y_dataerr))

    # Post-training evaluation and rendering:
    rng = jax.random.PRNGKey(0)
    rollout = []
    n_episodes = 1

    eval_env = env_maker()
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    jit_inference_fn = jax.jit(make_inference_fn(trained_params, deterministic=True))

    for _ in tqdm(range(n_episodes), total=n_episodes, desc="Collecting rollout"):
        state = jit_reset(rng)
        rollout.append(state)
        for i in range(eval_env._config.episode_length):
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, ctrl)
            rollout.append(state)
    render_every = 1
    frames = eval_env.render(rollout[::render_every])
    rewards = [s.reward for s in rollout]
    out_path = f"{args.output_dir}/rollout_{env_name}.mp4"
    media.write_video(out_path, np.array(frames), fps=1.0 / eval_env.dt / render_every)
    print(f"mean reward: {np.mean(rewards)}")
