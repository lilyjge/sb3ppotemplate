"""
Training Script for SB3 PPO Self-Driving Car Agent

This script sets up and trains a PPO agent using Stable-Baselines3.
Students should fill in the hyperparameters and training configuration.

Questions to consider:
- What hyperparameters affect learning speed and stability?
- How do you know if training is working?
- What's the difference between learning rate and batch size?
- How long should training run?
"""

import os
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from environment import SelfDrivingCarEnv


def create_env(config_path: str, render_mode: str = None):
    """
    Create and wrap the environment.
    
    Questions:
    - Why do we wrap environments in VecEnv?
    - What's the difference between DummyVecEnv and SubprocVecEnv?
    - When would you use multiple parallel environments?
    
    Args:
        config_path: Path to environment config file
        render_mode: Rendering mode (None for training)
        
    Returns:
        env: Wrapped environment
    """
    def _make_env():
        env = SelfDrivingCarEnv(config_path=config_path, render_mode=render_mode)
        # TODO: Wrap environment with Monitor for logging
        # What does Monitor do? Why is it useful?
        # env = Monitor(env, ???)
        return env
    
    # TODO: Create vectorized environment
    # Why use DummyVecEnv? When would you use SubprocVecEnv?
    # env = DummyVecEnv([_make_env])
    
    return None  # Placeholder


def load_config(config_path: str = "setup/track_config (1).yaml"):
    """
    Load training configuration.
    
    Questions:
    - What hyperparameters should be configurable?
    - Should you hardcode values or use config files?
    
    Args:
        config_path: Path to config file
        
    Returns:
        config: Configuration dictionary
    """
    # TODO: Load config file or define hyperparameters
    # Should you add training hyperparameters to the YAML file?
    # Or define them here as constants?
    
    config = {
        # TODO: Define PPO hyperparameters
        # What values should these have? How do you choose?
        # "learning_rate": ???,  # How fast should the agent learn?
        # "n_steps": ???,  # How many steps before updating?
        # "batch_size": ???,  # How many samples per update?
        # "n_epochs": ???,  # How many times to use the same data?
        # "gamma": ???,  # Discount factor - how much do future rewards matter?
        # "gae_lambda": ???,  # GAE parameter - what does this control?
        # "clip_range": ???,  # PPO clipping - why is this important?
        # "ent_coef": ???,  # Entropy coefficient - exploration vs exploitation
        # "vf_coef": ???,  # Value function coefficient
        # "max_grad_norm": ???,  # Gradient clipping - why use this?
        
        # TODO: Define training settings
        # "total_timesteps": ???,  # How long to train?
        # "eval_freq": ???,  # How often to evaluate?
        # "n_eval_episodes": ???,  # How many episodes for evaluation?
    }
    
    return config


def main():
    """
    Main training function.
    
    Questions:
    - How do you initialize a PPO agent?
    - What callbacks are useful during training?
    - How do you save and load models?
    - What metrics should you track?
    """
    
    # TODO: Set up paths
    # Where should models be saved? Logs?
    # models_dir = "models/"
    # logs_dir = "logs/"
    # os.makedirs(models_dir, exist_ok=True)
    # os.makedirs(logs_dir, exist_ok=True)
    
    # TODO: Load configuration
    # config = load_config()
    
    # TODO: Create training environment
    # What render mode should you use for training? Why?
    # train_env = create_env(???, render_mode=???)
    
    # TODO: Create evaluation environment
    # Should evaluation use a different render mode?
    # eval_env = create_env(???, render_mode=???)
    
    # TODO: Initialize PPO agent
    # What parameters does PPO need?
    # Think about: policy type, environment, learning rate, etc.
    # model = PPO(
    #     policy="MlpPolicy",  # What other policies are available?
    #     env=???,
    #     learning_rate=???,
    #     n_steps=???,
    #     batch_size=???,
    #     n_epochs=???,
    #     gamma=???,
    #     gae_lambda=???,
    #     clip_range=???,
    #     ent_coef=???,
    #     vf_coef=???,
    #     max_grad_norm=???,
    #     verbose=1,  # What does verbose do?
    #     tensorboard_log=logs_dir,  # How do you view tensorboard logs?
    # )
    
    # TODO: Set up callbacks
    # What callbacks are useful?
    # - EvalCallback: Evaluate during training
    # - CheckpointCallback: Save model checkpoints
    # callbacks = []
    
    # TODO: Create evaluation callback
    # How often should you evaluate? What metrics matter?
    # eval_callback = EvalCallback(
    #     eval_env,
    #     best_model_save_path=???,
    #     log_path=???,
    #     eval_freq=???,
    #     deterministic=True,  # Should evaluation be deterministic?
    #     render=False,  # Should you render during evaluation?
    # )
    # callbacks.append(eval_callback)
    
    # TODO: Create checkpoint callback
    # How often should you save checkpoints?
    # checkpoint_callback = CheckpointCallback(
    #     save_freq=???,
    #     save_path=???,
    #     name_prefix="ppo_car",
    # )
    # callbacks.append(checkpoint_callback)
    
    # TODO: Train the model
    # How do you start training?
    # What does learn() return?
    # model.learn(
    #     total_timesteps=???,
    #     callback=callbacks,
    #     progress_bar=True,  # Should you show progress bar?
    # )
    
    # TODO: Save final model
    # Where should the final model be saved?
    # model.save(???)
    
    # TODO: Clean up
    # What needs to be closed?
    # train_env.close()
    # eval_env.close()
    
    print("Training complete! Check the models/ directory for saved models.")


if __name__ == "__main__":
    main()

