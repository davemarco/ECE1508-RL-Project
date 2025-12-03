import os
import sys

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"
else:
    os.environ["MUJOCO_GL"] = "glfw"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"

import csv
import json
import random
import time
import math
import warnings
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import tqdm
import wandb
import numpy as np

try:
    # Required for avoiding IsaacGym import error
    import isaacgym
except ImportError:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler

from tensordict import TensorDict

from fast_td3.fast_td3_utils import (
    EmpiricalNormalization,
    RewardNormalizer,
    PerTaskRewardNormalizer,
    SimpleReplayBuffer,
    save_params,
    mark_step,
)
from fast_td3.hyperparams import get_args

torch.set_float32_matmul_precision("high")

try:
    import jax.numpy as jnp
except ImportError:
    pass

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

from core.envs.mujoco_playground_env_humanoid_uneven import make_env

from tensordict import from_module


def configure_amp(args):
    amp_enabled = args.amp and args.cuda and torch.cuda.is_available()
    amp_device_type = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float32
    scaler = GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
    return amp_enabled, amp_device_type, amp_dtype, scaler


def enforce_cuda_device(device_rank: int):
    if not torch.cuda.is_available():
        raise ValueError("No GPU available")
    device = torch.device(f"cuda:{device_rank}")
    print(f"Using device: {device}")
    return device


def seed_everything(seed: int, deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic


def prepare_output_dirs(output_dir: str, run_name: str):
    base_dir = Path(output_dir).expanduser().resolve()
    run_dir = base_dir / run_name
    checkpoints_dir = run_dir / "checkpoints"
    render_dir = run_dir / "renders"

    for path in (base_dir, run_dir, checkpoints_dir, render_dir):
        path.mkdir(parents=True, exist_ok=True)

    return run_dir, checkpoints_dir, render_dir


def derive_env_dimensions(envs):
    n_act = envs.num_actions
    n_obs = envs.num_obs if isinstance(envs.num_obs, int) else envs.num_obs[0]
    if envs.asymmetric_obs:
        n_critic_obs = (
            envs.num_privileged_obs
            if isinstance(envs.num_privileged_obs, int)
            else envs.num_privileged_obs[0]
        )
    else:
        n_critic_obs = n_obs
    return n_obs, n_critic_obs, n_act


def maybe_save_render_video(
    renders,
    render_dir: Path,
    run_name: str,
    global_step: int,
    output_path: Optional[Path] = None,
):
    if renders is None or len(renders) == 0:
        return None

    frames = np.asarray(renders)
    if frames.size == 0:
        return None

    frames = np.clip(frames, 0, 255).astype(np.uint8)
    render_dir.mkdir(parents=True, exist_ok=True)
    if output_path is not None:
        video_path = Path(output_path)
    else:
        video_path = render_dir / f"{run_name}_step{global_step}.gif"

    if imageio is not None:
        imageio.mimsave(
            video_path,
            list(frames),
            format="GIF",
            duration=1 / 30.0,
        )
        return video_path

    fallback_path = video_path.with_suffix(".npz")
    np.savez_compressed(fallback_path, frames=frames)
    print(
        f"[FastTD3] imageio not available; saved raw render frames to {fallback_path}"
    )
    return fallback_path


def to_scalar(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return float(value.detach().cpu().mean().item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class CSVLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.file = self.path.open("w", newline="")
        self.fieldnames = None
        self.writer = None

    def write_row(self, row: dict):
        if self.fieldnames is None:
            keys = [k for k in row.keys() if k != "step"]
            self.fieldnames = ["step"] + sorted(keys)
            self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
            self.writer.writeheader()
        stable_row = {k: row.get(k, "") for k in self.fieldnames}
        self.writer.writerow(stable_row)
        self.file.flush()

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass


class LossPlotter:
    DEFAULT_KEYS = [
        "actor_loss",
        "qf_loss",
        "actor_grad_norm",
        "critic_grad_norm",
    ]

    def __init__(self, path: Path, keys=None):
        self.path = Path(path)
        self.keys = keys or self.DEFAULT_KEYS
        self.data = {k: [] for k in self.keys}
        self.steps = []

    def update(self, step: int, metrics: dict):
        self.steps.append(step)
        for k in self.keys:
            self.data[k].append(to_scalar(metrics.get(k)))

    def save(self):
        if not self.steps:
            return
        try:
            plt.figure(figsize=(10, 6))
            plotted = 0
            for k, series in self.data.items():
                arr = np.array(series, dtype=float)
                if np.isfinite(arr).any():
                    plt.plot(self.steps, arr, label=k)
                    plotted += 1
            if plotted == 0:
                plt.close()
                return
            plt.xlabel("global step")
            plt.ylabel("value")
            plt.title("Training diagnostics")
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(self.path)
            plt.close()
        except Exception as exc:
            warnings.warn(f"Failed to save loss plot: {exc}")


class RewardPlotter:
    def __init__(self, path: Path, keys=None):
        self.path = Path(path)
        self.primary_keys = keys or ["env_rewards", "buffer_rewards", "eval_avg_return"]
        self.steps = []
        self.data = {k: [] for k in self.primary_keys}

    def update(self, step: int, metrics: dict):
        self.steps.append(step)
        for k in self.primary_keys:
            self.data[k].append(to_scalar(metrics.get(k)))

    def save(self):
        if not self.steps:
            return
        try:
            plt.figure(figsize=(10, 6))
            plotted = 0
            for key, series in self.data.items():
                arr = np.array(series, dtype=float)
                if np.isfinite(arr).any():
                    plt.plot(self.steps, arr, label=key)
                    plotted += 1
            if plotted == 0:
                plt.close()
                return
            plt.xlabel("global step")
            plt.ylabel("reward")
            plt.title("Reward metrics")
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(self.path)
            plt.close()
        except Exception as exc:
            warnings.warn(f"Failed to save reward plot: {exc}")


def save_hyperparameters(args, output_path: Path):
    args_dict = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            args_dict[key] = str(value)
        else:
            args_dict[key] = value
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(args_dict, f, indent=2, sort_keys=True)


def main():
    args = get_args()
    print(args)
    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}"
    run_dir, checkpoints_dir, render_dir = prepare_output_dirs(
        args.output_dir, run_name
    )
    print(f"Artifacts directory: {run_dir}")
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_logger = CSVLogger(metrics_dir / "metrics.csv")
    loss_plotter = LossPlotter(metrics_dir / "losses.png")
    reward_plotter = RewardPlotter(metrics_dir / "progress.png")
    latest_checkpoint_path = checkpoints_dir / f"{run_name}_latest.pt"
    latest_render_path = render_dir / f"{run_name}_latest.gif"
    save_hyperparameters(args, run_dir / "hyperparameters.json")

    amp_enabled, amp_device_type, amp_dtype, scaler = configure_amp(args)

    if args.use_wandb:
        wandb.init(
            project=args.project,
            name=run_name,
            config=vars(args),
            save_code=True,
            dir=str(run_dir),
        )

    seed_everything(args.seed, args.torch_deterministic)
    device = enforce_cuda_device(args.device_rank)

    env_type = "mujoco_playground"
    envs, eval_envs, render_env = make_env(
        args.env_name,
        args.seed,
        args.num_envs,
        args.num_eval_envs,
        args.device_rank,
        use_tuned_reward=args.use_tuned_reward,
        use_domain_randomization=args.use_domain_randomization,
        use_push_randomization=args.use_push_randomization,
    )

    n_obs, n_critic_obs, n_act = derive_env_dimensions(envs)
    num_tasks = int(getattr(envs, "num_tasks", 0) or 0)
    use_task_embedding = num_tasks > 0
    action_low, action_high = -1.0, 1.0

    if args.obs_normalization:
        obs_normalizer = EmpiricalNormalization(shape=n_obs, device=device)
        critic_obs_normalizer = EmpiricalNormalization(
            shape=n_critic_obs, device=device
        )
    else:
        obs_normalizer = nn.Identity()
        critic_obs_normalizer = nn.Identity()

    if args.reward_normalization:
        if use_task_embedding:
            reward_normalizer = PerTaskRewardNormalizer(
                num_tasks=num_tasks,
                gamma=args.gamma,
                device=device,
                g_max=min(abs(args.v_min), abs(args.v_max)),
            )
        else:
            reward_normalizer = RewardNormalizer(
                gamma=args.gamma,
                device=device,
                g_max=min(abs(args.v_min), abs(args.v_max)),
            )
    else:
        reward_normalizer = nn.Identity()

    actor_kwargs = {
        "n_obs": n_obs,
        "n_act": n_act,
        "num_envs": args.num_envs,
        "device": device,
        "init_scale": args.init_scale,
        "hidden_dim": args.actor_hidden_dim,
        "std_min": args.std_min,
        "std_max": args.std_max,
    }
    critic_kwargs = {
        "n_obs": n_critic_obs,
        "n_act": n_act,
        "num_atoms": args.num_atoms,
        "v_min": args.v_min,
        "v_max": args.v_max,
        "hidden_dim": args.critic_hidden_dim,
        "device": device,
    }

    if use_task_embedding:
        actor_kwargs["n_obs"] = n_obs - num_tasks + args.task_embedding_dim
        critic_kwargs["n_obs"] = n_critic_obs - num_tasks + args.task_embedding_dim
        actor_kwargs["num_tasks"] = num_tasks
        actor_kwargs["task_embedding_dim"] = args.task_embedding_dim
        critic_kwargs["num_tasks"] = num_tasks
        critic_kwargs["task_embedding_dim"] = args.task_embedding_dim

    if args.agent == "fasttd3":
        if use_task_embedding:
            from fast_td3.fast_td3 import MultiTaskActor as Actor, MultiTaskCritic as Critic
        else:
            from fast_td3.fast_td3 import Actor, Critic
        actor_cls = Actor
        critic_cls = Critic
        print("Using FastTD3")
    elif args.agent == "fasttd3_simbav2":
        if use_task_embedding:
            from fast_td3.fast_td3_simbav2 import (
                MultiTaskActor as Actor,
                MultiTaskCritic as Critic,
            )
        else:
            from fast_td3.fast_td3_simbav2 import Actor, Critic
        actor_cls = Actor
        critic_cls = Critic
        print("Using FastTD3 + SimbaV2")
        actor_kwargs.pop("init_scale")
        actor_kwargs.update(
            {
                "scaler_init": math.sqrt(2.0 / args.actor_hidden_dim),
                "scaler_scale": math.sqrt(2.0 / args.actor_hidden_dim),
                "alpha_init": 1.0 / (args.actor_num_blocks + 1),
                "alpha_scale": 1.0 / math.sqrt(args.actor_hidden_dim),
                "expansion": 4,
                "c_shift": 3.0,
                "num_blocks": args.actor_num_blocks,
            }
        )
        critic_kwargs.update(
            {
                "scaler_init": math.sqrt(2.0 / args.critic_hidden_dim),
                "scaler_scale": math.sqrt(2.0 / args.critic_hidden_dim),
                "alpha_init": 1.0 / (args.critic_num_blocks + 1),
                "alpha_scale": 1.0 / math.sqrt(args.critic_hidden_dim),
                "num_blocks": args.critic_num_blocks,
                "expansion": 4,
                "c_shift": 3.0,
            }
        )
    else:
        raise ValueError(f"Agent {args.agent} not supported")

    actor = actor_cls(**actor_kwargs)

    actor_detach = actor_cls(**actor_kwargs)
    # Copy params to actor_detach without grad
    from_module(actor).data.to_module(actor_detach)
    policy = actor_detach.explore

    qnet = critic_cls(**critic_kwargs)
    qnet_target = critic_cls(**critic_kwargs)
    qnet_target.load_state_dict(qnet.state_dict())

    q_optimizer = optim.AdamW(
        list(qnet.parameters()),
        lr=torch.tensor(args.critic_learning_rate, device=device),
        weight_decay=args.weight_decay,
    )
    actor_optimizer = optim.AdamW(
        list(actor.parameters()),
        lr=torch.tensor(args.actor_learning_rate, device=device),
        weight_decay=args.weight_decay,
    )

    # Add learning rate schedulers
    q_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        q_optimizer,
        T_max=args.total_timesteps,
        eta_min=torch.tensor(args.critic_learning_rate_end, device=device),
    )
    actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        actor_optimizer,
        T_max=args.total_timesteps,
        eta_min=torch.tensor(args.actor_learning_rate_end, device=device),
    )

    rb = SimpleReplayBuffer(
        n_env=args.num_envs,
        buffer_size=args.buffer_size,
        n_obs=n_obs,
        n_act=n_act,
        n_critic_obs=n_critic_obs,
        asymmetric_obs=envs.asymmetric_obs,
        playground_mode=env_type == "mujoco_playground",
        n_steps=args.num_steps,
        gamma=args.gamma,
        device=device,
    )

    policy_noise = args.policy_noise
    noise_clip = args.noise_clip

    def evaluate():
        num_eval_envs = eval_envs.num_envs
        episode_returns = torch.zeros(num_eval_envs, device=device)
        episode_lengths = torch.zeros(num_eval_envs, device=device)
        done_masks = torch.zeros(num_eval_envs, dtype=torch.bool, device=device)

        if env_type == "isaaclab":
            obs = eval_envs.reset(random_start_init=False)
        else:
            obs = eval_envs.reset()

        # Run for a fixed number of steps
        for i in range(eval_envs.max_episode_steps):
            with torch.no_grad(), autocast(
                device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
            ):
                obs = normalize_obs(obs, update=False)
                actions = actor(obs)

            next_obs, rewards, dones, infos = eval_envs.step(actions.float())

            episode_returns = torch.where(
                ~done_masks, episode_returns + rewards, episode_returns
            )
            episode_lengths = torch.where(
                ~done_masks, episode_lengths + 1, episode_lengths
            )

            done_masks = torch.logical_or(done_masks, dones)
            if done_masks.all():
                break
            obs = next_obs

        return (
            episode_returns.mean().item(),
            episode_returns.std().item(),
            episode_lengths.mean().item(),
        )

    def render_with_rollout():
        # Quick rollout for rendering
        obs = render_env.reset()
        render_env.state.info["command"] = jnp.array([[1.0, 0.0, 0.0]])
        renders = [render_env.state]
        for i in range(render_env.max_episode_steps):
            with torch.no_grad(), autocast(
                device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
            ):
                obs = normalize_obs(obs, update=False)
                actions = actor(obs)
            next_obs, _, done, _ = render_env.step(actions.float())
            render_env.state.info["command"] = jnp.array([[1.0, 0.0, 0.0]])
            if i % 2 == 0:
                renders.append(render_env.state)
            if done.any():
                break
            obs = next_obs

        renders = render_env.render_trajectory(renders)
        return renders

    def update_main(data, logs_dict):
        with autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            observations = data["observations"]
            next_observations = data["next"]["observations"]
            if envs.asymmetric_obs:
                critic_observations = data["critic_observations"]
                next_critic_observations = data["next"]["critic_observations"]
            else:
                critic_observations = observations
                next_critic_observations = next_observations
            actions = data["actions"]
            rewards = data["next"]["rewards"]
            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            if args.disable_bootstrap:
                bootstrap = (~dones).float()
            else:
                bootstrap = (truncations | ~dones).float()

            clipped_noise = torch.randn_like(actions)
            clipped_noise = clipped_noise.mul(policy_noise).clamp(
                -noise_clip, noise_clip
            )

            next_state_actions = (actor(next_observations) + clipped_noise).clamp(
                action_low, action_high
            )
            discount = args.gamma ** data["next"]["effective_n_steps"]

            with torch.no_grad():
                qf1_next_target_projected, qf2_next_target_projected = (
                    qnet_target.projection(
                        next_critic_observations,
                        next_state_actions,
                        rewards,
                        bootstrap,
                        discount,
                    )
                )
                qf1_next_target_value = qnet_target.get_value(qf1_next_target_projected)
                qf2_next_target_value = qnet_target.get_value(qf2_next_target_projected)
                if args.use_cdq:
                    qf_next_target_dist = torch.where(
                        qf1_next_target_value.unsqueeze(1)
                        < qf2_next_target_value.unsqueeze(1),
                        qf1_next_target_projected,
                        qf2_next_target_projected,
                    )
                    qf1_next_target_dist = qf2_next_target_dist = qf_next_target_dist
                else:
                    qf1_next_target_dist, qf2_next_target_dist = (
                        qf1_next_target_projected,
                        qf2_next_target_projected,
                    )

            qf1, qf2 = qnet(critic_observations, actions)
            qf1_loss = -torch.sum(
                qf1_next_target_dist * F.log_softmax(qf1, dim=1), dim=1
            ).mean()
            qf2_loss = -torch.sum(
                qf2_next_target_dist * F.log_softmax(qf2, dim=1), dim=1
            ).mean()
            qf_loss = qf1_loss + qf2_loss

        q_optimizer.zero_grad(set_to_none=True)
        scaler.scale(qf_loss).backward()
        scaler.unscale_(q_optimizer)

        if args.use_grad_norm_clipping:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                qnet.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            critic_grad_norm = torch.tensor(0.0, device=device)
        scaler.step(q_optimizer)
        scaler.update()

        logs_dict["critic_grad_norm"] = critic_grad_norm.detach()
        logs_dict["qf_loss"] = qf_loss.detach()
        logs_dict["qf_max"] = qf1_next_target_value.max().detach()
        logs_dict["qf_min"] = qf1_next_target_value.min().detach()
        return logs_dict

    def update_pol(data, logs_dict):
        with autocast(
            device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
        ):
            critic_observations = (
                data["critic_observations"]
                if envs.asymmetric_obs
                else data["observations"]
            )

            qf1, qf2 = qnet(critic_observations, actor(data["observations"]))
            qf1_value = qnet.get_value(F.softmax(qf1, dim=1))
            qf2_value = qnet.get_value(F.softmax(qf2, dim=1))
            if args.use_cdq:
                qf_value = torch.minimum(qf1_value, qf2_value)
            else:
                qf_value = (qf1_value + qf2_value) / 2.0
            actor_loss = -qf_value.mean()

        actor_optimizer.zero_grad(set_to_none=True)
        scaler.scale(actor_loss).backward()
        scaler.unscale_(actor_optimizer)
        if args.use_grad_norm_clipping:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                max_norm=args.max_grad_norm if args.max_grad_norm > 0 else float("inf"),
            )
        else:
            actor_grad_norm = torch.tensor(0.0, device=device)
        scaler.step(actor_optimizer)
        scaler.update()
        logs_dict["actor_grad_norm"] = actor_grad_norm.detach()
        logs_dict["actor_loss"] = actor_loss.detach()
        return logs_dict

    @torch.no_grad()
    def soft_update(src, tgt, tau: float):
        src_ps = [p.data for p in src.parameters()]
        tgt_ps = [p.data for p in tgt.parameters()]

        torch._foreach_mul_(tgt_ps, 1.0 - tau)
        torch._foreach_add_(tgt_ps, src_ps, alpha=tau)

    if args.compile:
        compile_mode = args.compile_mode
        update_main = torch.compile(update_main, mode=compile_mode)
        update_pol = torch.compile(update_pol, mode=compile_mode)
        policy = torch.compile(policy, mode=None)
        normalize_obs = torch.compile(obs_normalizer.forward, mode=None)
        normalize_critic_obs = torch.compile(critic_obs_normalizer.forward, mode=None)
        if args.reward_normalization:
            update_stats = torch.compile(reward_normalizer.update_stats, mode=None)
        normalize_reward = torch.compile(reward_normalizer.forward, mode=None)
    else:
        normalize_obs = obs_normalizer.forward
        normalize_critic_obs = critic_obs_normalizer.forward
        if args.reward_normalization:
            update_stats = reward_normalizer.update_stats
        normalize_reward = reward_normalizer.forward

    if envs.asymmetric_obs:
        obs, critic_obs = envs.reset_with_critic_obs()
        critic_obs = torch.as_tensor(critic_obs, device=device, dtype=torch.float)
    else:
        obs = envs.reset()
    if args.checkpoint_path:
        # Load checkpoint if specified
        torch_checkpoint = torch.load(
            f"{args.checkpoint_path}", map_location=device, weights_only=False
        )
        actor.load_state_dict(torch_checkpoint["actor_state_dict"])
        obs_normalizer.load_state_dict(torch_checkpoint["obs_normalizer_state"])
        critic_obs_normalizer.load_state_dict(
            torch_checkpoint["critic_obs_normalizer_state"]
        )
        qnet.load_state_dict(torch_checkpoint["qnet_state_dict"])
        qnet_target.load_state_dict(torch_checkpoint["qnet_target_state_dict"])
        global_step = torch_checkpoint["global_step"]
    else:
        global_step = 0

    dones = None
    pbar = tqdm.tqdm(total=args.total_timesteps, initial=global_step)
    start_time = None
    desc = ""
    
    x_data, y_data, y_dataerr = [], [], []

    try:
        while global_step < args.total_timesteps:
            mark_step()
            logs_dict = TensorDict()
            if (
                start_time is None
                and global_step >= args.measure_burnin + args.learning_starts
            ):
                start_time = time.time()
                measure_burnin = global_step

            with torch.no_grad(), autocast(
                device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enabled
            ):
                norm_obs = normalize_obs(obs)
                actions = policy(obs=norm_obs, dones=dones)

            next_obs, rewards, dones, infos = envs.step(actions.float())
            truncations = infos["time_outs"]

            if args.reward_normalization:
                if use_task_embedding:
                    task_ids = torch.argmax(obs[..., -num_tasks:], dim=1)
                    update_stats(rewards, dones.float(), task_ids)
                else:
                    update_stats(rewards, dones.float())

            observation_infos = infos["observations"]
            raw_observations = (
                observation_infos.get("raw")
                if hasattr(observation_infos, "get")
                else None
            )
            raw_obs = (
                raw_observations.get("obs")
                if raw_observations is not None and hasattr(raw_observations, "get")
                else None
            )

            if envs.asymmetric_obs:
                next_critic_obs = observation_infos["critic"]
                raw_critic_obs = (
                    raw_observations.get("critic_obs")
                    if raw_observations is not None
                    and hasattr(raw_observations, "get")
                    else None
                )

            # Compute 'true' next_obs and next_critic_obs for saving
            true_next_obs = (
                torch.where(dones[:, None] > 0, raw_obs, next_obs)
                if raw_obs is not None
                else next_obs
            )
            if envs.asymmetric_obs:
                true_next_critic_obs = (
                    torch.where(dones[:, None] > 0, raw_critic_obs, next_critic_obs)
                    if raw_critic_obs is not None
                    else next_critic_obs
                )

            transition = TensorDict(
                {
                    "observations": obs,
                    "actions": torch.as_tensor(actions, device=device, dtype=torch.float),
                    "next": {
                        "observations": true_next_obs,
                        "rewards": torch.as_tensor(
                            rewards, device=device, dtype=torch.float
                        ),
                        "truncations": truncations.long(),
                        "dones": dones.long(),
                    },
                },
                batch_size=(envs.num_envs,),
                device=device,
            )
            if envs.asymmetric_obs:
                transition["critic_observations"] = critic_obs
                transition["next"]["critic_observations"] = true_next_critic_obs
            rb.extend(transition)

            obs = next_obs
            if envs.asymmetric_obs:
                critic_obs = next_critic_obs

            if global_step > args.learning_starts:
                for i in range(args.num_updates):
                    data = rb.sample(max(1, args.batch_size // args.num_envs))
                    data["observations"] = normalize_obs(data["observations"])
                    data["next"]["observations"] = normalize_obs(
                        data["next"]["observations"]
                    )
                    if envs.asymmetric_obs:
                        data["critic_observations"] = normalize_critic_obs(
                            data["critic_observations"]
                        )
                        data["next"]["critic_observations"] = normalize_critic_obs(
                            data["next"]["critic_observations"]
                        )
                    raw_rewards = data["next"]["rewards"]
                    if use_task_embedding and args.reward_normalization:
                        # Multi-task reward normalization
                        task_ids_one_hot = data["observations"][..., -num_tasks :]
                        task_indices = torch.argmax(task_ids_one_hot, dim=1)
                        data["next"]["rewards"] = normalize_reward(
                            raw_rewards, task_ids=task_indices
                        )
                    else:
                        data["next"]["rewards"] = normalize_reward(raw_rewards)

                    logs_dict = update_main(data, logs_dict)
                    if args.num_updates > 1:
                        if i % args.policy_frequency == 1:
                            logs_dict = update_pol(data, logs_dict)
                    else:
                        if global_step % args.policy_frequency == 0:
                            logs_dict = update_pol(data, logs_dict)

                    soft_update(qnet, qnet_target, args.tau)

                if global_step % 100 == 0 and start_time is not None:
                    speed = (global_step - measure_burnin) / (
                        time.time() - start_time
                    )
                    pbar.set_description(f"{speed: 4.4f} sps, " + desc)
                    with torch.no_grad():
                        logs = {
                            "actor_loss": logs_dict["actor_loss"].mean(),
                            "qf_loss": logs_dict["qf_loss"].mean(),
                            "qf_max": logs_dict["qf_max"].mean(),
                            "qf_min": logs_dict["qf_min"].mean(),
                            "actor_grad_norm": logs_dict["actor_grad_norm"].mean(),
                            "critic_grad_norm": logs_dict["critic_grad_norm"].mean(),
                            "env_rewards": rewards.mean(),
                            "buffer_rewards": raw_rewards.mean(),
                        }

                        if (
                            args.eval_interval > 0
                            and global_step % args.eval_interval == 0
                        ):
                            print(f"Evaluating at global step {global_step}")
                            eval_avg_return, eval_avg_std, eval_avg_length = evaluate()
                            if env_type in ["humanoid_bench", "isaaclab", "mtbench"]:
                                # NOTE: Hacky way of evaluating performance, but just works
                                obs = envs.reset()
                            logs["eval_avg_return"] = eval_avg_return
                            logs["eval_return_std"] = eval_avg_std
                            logs["eval_avg_length"] = eval_avg_length
                            
                            x_data.append(global_step * args.num_envs)
                            y_data.append(eval_avg_return)
                            y_dataerr.append(eval_avg_std)

                        if (
                            args.render_interval > 0
                            and global_step % args.render_interval == 0
                        ):
                            renders = render_with_rollout()
                            render_frames = np.asarray(renders)
                            saved_render_path = maybe_save_render_video(
                                render_frames,
                                render_dir,
                                run_name,
                                global_step,
                                output_path=latest_render_path,
                            )
                            if args.use_wandb:
                                render_video = wandb.Video(
                                    render_frames.transpose(
                                        0, 3, 1, 2
                                    ),  # Convert to (T, C, H, W) format
                                    fps=30,
                                    format="gif",
                                )
                                logs["render_video"] = render_video
                            if saved_render_path is not None:
                                logs["render_video_path"] = str(saved_render_path)
                                print(f"Saved render to {saved_render_path}")

                        scalar_logs = {k: to_scalar(v) for k, v in logs.items()}
                        scalar_logs.update(
                            {
                                "step": global_step,
                                "speed": to_scalar(speed),
                                "frame": global_step * args.num_envs,
                                "critic_lr": to_scalar(q_scheduler.get_last_lr()[0]),
                                "actor_lr": to_scalar(actor_scheduler.get_last_lr()[0]),
                            }
                        )
                        csv_logger.write_row(scalar_logs)
                        loss_plotter.update(global_step, scalar_logs)
                        loss_plotter.save()
                        reward_plotter.update(global_step, scalar_logs)
                        reward_plotter.save()

                    if args.use_wandb:
                        wandb.log(
                            {
                                "speed": speed,
                                "frame": global_step * args.num_envs,
                                "critic_lr": q_scheduler.get_last_lr()[0],
                                "actor_lr": actor_scheduler.get_last_lr()[0],
                                **logs,
                            },
                            step=global_step,
                        )

                if (
                    args.save_interval > 0
                    and global_step > 0
                    and global_step % args.save_interval == 0
                ):
                    print(f"Saving model at global step {global_step}")
                    save_params(
                        global_step,
                        actor,
                        qnet,
                        qnet_target,
                        obs_normalizer,
                        critic_obs_normalizer,
                        args,
                        str(latest_checkpoint_path),
                    )

            global_step += 1
            actor_scheduler.step()
            q_scheduler.step()
            pbar.update(1)
    finally:
        csv_logger.close()
        
        # Save curves arrays
        with open(f"{run_dir}/x_data_{args.agent}_{args.env_name}.npy", "wb") as f:
            np.save(f, np.array(x_data))
        with open(f"{run_dir}/y_data_{args.agent}_{args.env_name}.npy", "wb") as f:
            np.save(f, np.array(y_data))
        with open(f"{run_dir}/y_dataerr_{args.agent}_{args.env_name}.npy", "wb") as f:
            np.save(f, np.array(y_dataerr))

    save_params(
        global_step,
        actor,
        qnet,
        qnet_target,
        obs_normalizer,
        critic_obs_normalizer,
        args,
        str(latest_checkpoint_path),
    )


if __name__ == "__main__":
    main()
