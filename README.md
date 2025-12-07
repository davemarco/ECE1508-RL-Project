# Installation

To use this repo first clone it with the command 
```
git clone https://github.com/davemarco/ECE1508-RL-Project
```

Set up the environment with the command
```
pip3 install -r requirements.txt
```

# Training & Evaluation

## PPO
To train PPO in the baseline environment, run this command
```
python3 train_PPO_SAC.py --algo PPO --env HumanoidWalk --config_file PPO_config.json \
--output_dir <path_to_output_directory>
```
To train PPO in the modified environment, run this command
```
python3 train_PPO_SAC.py --algo PPO --env HumanoidWalkWithObstacles --config_file PPO_config.json \
--output_dir <path_to_output_directory>
```

## SAC
To train SAC in the baseline environment, run this command
```
python3 train_PPO_SAC.py --algo SAC --env HumanoidWalk --config_file SAC_config.json \
--output_dir <path_to_output_directory>
```
To train SAC in the modified environment, run this command
```
python3 train_PPO_SAC.py --algo SAC --env HumanoidWalkWithObstacles --config_file SAC_config.json \
--output_dir <path_to_output_directory>
```


## TD3 (FastTD3)
To train TD3 in the baseline environment, use this command
```
python3 train_fasttd3.py --env_name HumanoidWalk \
--exp_name <experiment_name> \
--render_interval 10000 \
--agent fasttd3_simbav2 \
--batch_size 2048 \
--critic_learning_rate 3e-5 \
--actor_learning_rate 3e-5 \
--critic_learning_rate_end 3e-6 \
--actor_learning_rate_end 3e-6 \
--weight_decay 0.0 \
--total_timesteps 500000 \
--std_max 0.25 \
--v_min -250.0 \
--v_max 250.0 \
--num_envs 1024 \
--num_steps 2 \
--num_updates 1 \
--policy_noise 0.1 \
--noise_clip 0.5 \
--policy_frequency 2 \
--max_grad_norm 0.5 \
--tau 0.002 \
--buffer_size 10000 \
--learning_starts 10000 \
--gamma 0.97 \
--seed <seed> \
--use_wandb \
--output_dir <path_to_output_directory>
```

To train TD3 in the modified environment, use this command
```
python3 train_fasttd3_uneven.py --env_name HumanoidWalk \
--exp_name <experiment_name> \
--render_interval 10000 \
--agent fasttd3_simbav2 \
--batch_size 2048 \
--critic_learning_rate 3e-5 \
--actor_learning_rate 3e-5 \
--critic_learning_rate_end 3e-6 \
--actor_learning_rate_end 3e-6 \
--weight_decay 0.0 \
--total_timesteps 500000 \
--std_max 0.25 \
--v_min -250.0 \
--v_max 250.0 \
--num_envs 1024 \
--num_steps 2 \
--num_updates 1 \
--policy_noise 0.1 \
--noise_clip 0.5 \
--policy_frequency 2 \
--max_grad_norm 0.5 \
--tau 0.002 \
--buffer_size 10000 \
--learning_starts 10000 \
--gamma 0.97 \
--seed <seed> \
--use_wandb \
--output_dir <path_to_output_directory>
```


# Some notes
- The provided config files contain the hyperparameters for our most recent experiments. Due to the randomness of the training, the results may vary; training may even diverge.
- For FastTD3, you may modify the hyperparameters in the `fast_td3/hyperparams.py` file and also pass them as command line arguments.
