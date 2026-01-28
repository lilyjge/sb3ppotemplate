"""
Self-Driving Car Environment for SB3 PPO Training

This is a Gymnasium-compatible environment that uses PyBullet for physics simulation.
The environment integrates with the Track and TankDriveController classes from the setup folder.

TODO: Fill in the blanks marked with # TODO comments and answer the Socratic questions!
"""

import os
import numpy as np
import pybullet as p
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, Any
import yaml

# Import the track and controller from setup
from setup.track import Track
from setup.controls import TankDriveController


class SelfDrivingCarEnv(gym.Env):
    """
    A self-driving car environment using PyBullet physics.
    
    The agent controls a car that must navigate around a procedural racetrack.
    Think about: What makes a good observation space for a self-driving car?
    What actions should the agent be able to take?
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        config_path: str = "setup/track_config (1).yaml",
        render_mode: Optional[str] = None,
        max_episode_steps: int = 1000,
    ):
        """
        Initialize the environment.
        
        Questions to consider:
        - What information does the agent need to see? (observation space)
        - What actions can the agent take? (action space)
        - How do we define success or failure? (reward function)
        - When should an episode end? (termination conditions)
        
        Args:
            config_path: Path to the YAML configuration file
            render_mode: "human" for GUI, "rgb_array" for images, None for headless
            max_episode_steps: Maximum steps before truncation
        """
        super().__init__()
        
        self.config_path = config_path
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        
        # Load configuration
        # TODO: Load the YAML config file here. What information do you need from it?
        # Hint: You'll need physics settings, spawn position, etc.
        self.config = yaml.safe_load(open(self.config_path, "r"))
        
        # Initialize PyBullet
        # TODO: How do you connect to PyBullet? What's the difference between GUI and DIRECT mode?
        # Think about: When would you use each mode?
        if render_mode == "human":
            physics_client = p.connect(p.GUI)
        else:
            physics_client = p.connect(p.DIRECT)
        
        self.physics_client = physics_client
        
        # TODO: Set up physics parameters from config
        # What physics settings affect the simulation? (gravity, time step, etc.)
        p.setGravity(0, 0, -9.8, physicsClientId=self.physics_client)
        p.setTimeStep(self.config["physics"]["time_step"], physicsClientId=self.physics_client)
        
        # Initialize track
        # TODO: Create a Track instance. What parameters does it need?
        self.track = Track(self.config_path, self.config["track"]["seed"])
        self.track.spawn_in_pybullet(self.physics_client)
        
        # Load car URDF
        # TODO: Load the car model. Where is the URDF file located?
        # Think about: What happens if the file doesn't exist?
        car_urdf_path = "setup/car (1).urdf"
        self.car_id = p.loadURDF(car_urdf_path, physicsClientId=self.physics_client)
        
        # TODO: Set initial car position and orientation from config
        # What's the difference between position and orientation?
        spawn_pos = self.config["spawn"]["position"]
        spawn_orn = self.config["spawn"]["orientation"]
        p.resetBasePositionAndOrientation(self.car_id, spawn_pos, spawn_orn, physicsClientId=self.physics_client)
        
        # Initialize controller
        # TODO: Create the TankDriveController. What does it need?
        self.controller = TankDriveController(self.config_path, self.car_id, self.physics_client)
        
        # Define action space
        # TODO: What actions should the agent control?
        # Think about: The TankDriveController uses forward_input (-1 to +1) and turn_input (-1 to +1)
        # What type of action space is appropriate? (Discrete, Box, etc.)
        # Should actions be continuous or discrete? Why?
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        
        # Define observation space
        # TODO: What should the agent observe?
        # Consider: Car position, velocity, orientation, distance to track boundaries, etc.
        # What shape should observations have? What are reasonable bounds?
        self.observation_space = spaces.Box(low=np.array([-np.inf]*5 + [-1]* 4 + [0]* 2), 
                                            high=np.array([np.inf]*5 + [1]* 4 + [0.75]* 2),
                                            shape=(11,),)
        
        # TODO: Initialize any tracking variables you need
        # What information do you need to compute rewards and check termination?
        self.last_position = np.array(self.config["spawn"]["position"])
        self.progress = 0.0
        self.episode_reward = 0.0
        
    def _get_observation(self) -> np.ndarray:
        """
        Get the current observation.
        
        Questions to consider:
        - What information helps the agent make good decisions?
        - Should observations be normalized? Why or why not?
        - How can you extract useful features from the car's state?
        
        Returns:
            observation: The current state observation
        """
        # TODO: Get car's current state from PyBullet
        # What information can you get about the car's position, orientation, and velocity?
        car_pos, car_orn = p.getBasePositionAndOrientation(self.car_id, physicsClientId=self.physics_client)
        car_vel, car_ang_vel = p.getBaseVelocity(self.car_id, physicsClientId=self.physics_client)
        
        # TODO: Extract useful features
        # Think about:
        # - How can you represent orientation? (Euler angles, quaternion, rotation matrix?) -> quaternion
        # - What velocity components matter? (forward speed, angular velocity?) -> forward speed and angular velocity
        # - How can you measure distance to track boundaries?
        car_pos_array = np.array(car_pos)
        # Compute minimum distance to inner and outer boundaries
        distances_to_inner = np.linalg.norm(self.track.inner_points - car_pos_array, axis=1)
        distances_to_outer = np.linalg.norm(self.track.outer_points - car_pos_array, axis=1)
        distance_to_inner_boundary = np.min(distances_to_inner)
        distance_to_outer_boundary = np.min(distances_to_outer)
        # - Should you include relative position on the track?
        # Compute forward speed (magnitude of velocity)
        forward_speed = np.linalg.norm(car_vel)
        # Compute angular speed magnitude
        angular_speed = np.linalg.norm(car_ang_vel)
        # TODO: Combine features into observation vector
        # Observation: [car_pos (3), forward_speed (1), angular_speed (1), car_orn (4), distance_to_inner (1), distance_to_outer (1)] = 11 elements
        observation = np.array([*car_pos, forward_speed, angular_speed, *car_orn, distance_to_inner_boundary, distance_to_outer_boundary], dtype=np.float32)
        
        # TODO: Normalize observations if needed
        # Why might normalization help training?
        # Only normalize bounded elements (quaternion and distances), leave unbounded elements as-is
        low = self.observation_space.low
        high = self.observation_space.high
        # Normalize only where bounds are finite
        finite_mask = np.isfinite(low) & np.isfinite(high)
        observation[finite_mask] = (observation[finite_mask] - low[finite_mask]) / (high[finite_mask] - low[finite_mask])
        return observation
    
    def _compute_reward(self, action: np.ndarray) -> float:
        """
        Compute the reward for the current step.
        
        This is a critical function! The reward signal shapes what the agent learns.
        
        Questions to consider:
        - What behaviors do you want to encourage? (staying on track, making progress, etc.)
        - What behaviors should be penalized? (going off track, crashing, etc.)
        - How do you balance different objectives? (progress vs. safety)
        - Should rewards be sparse (only at milestones) or dense (every step)?
        
        Args:
            action: The action taken by the agent
            
        Returns:
            reward: The reward value
        """
        # TODO: Get current car state
        car_pos, car_orn = p.getBasePositionAndOrientation(self.car_id, physicsClientId=self.physics_client)
        car_vel, car_ang_vel = p.getBaseVelocity(self.car_id, physicsClientId=self.physics_client)
        
        # TODO: Check if car is on track
        # How can you determine if the car is within track boundaries?
        # Hint: Use self.track.inner_points and self.track.outer_points
        is_on_track = np.all(car_pos > self.track.inner_points) and np.all(car_pos < self.track.outer_points)
        
        # TODO: Calculate progress
        # How do you measure progress around the track?
        # Think about: Distance traveled, laps completed, distance from start, etc.
        progress_reward = np.linalg.norm(np.array(car_pos) - np.array(self.last_position))
        
        # TODO: Calculate speed reward
        # Should the agent be rewarded for speed? Why or why not?
        speed_reward = np.linalg.norm(car_vel)
        
        # TODO: Calculate penalty for going off track
        # How severe should the penalty be? Should it end the episode?
        off_track_penalty = -100 if not is_on_track else 0
        
        # TODO: Combine rewards
        # How do you weight different reward components?
        # What happens if rewards are too large or too small?
        reward = 100 * progress_reward + 10 * speed_reward + off_track_penalty
        
        return reward
    
    def _is_terminated(self) -> bool:
        """
        Check if the episode should terminate (failure condition).
        
        Questions:
        - What constitutes a failure? (off track, crashed, etc.)
        - Should termination be immediate or gradual?
        
        Returns:
            terminated: True if episode should end due to failure
        """
        # TODO: Check termination conditions
        # When should the episode end in failure?
        # Think about: Distance from track, car orientation, etc.
        car_pos, car_orn = p.getBasePositionAndOrientation(self.car_id, physicsClientId=self.physics_client)
        is_off_track = np.all(car_pos < self.track.inner_points) or np.all(car_pos > self.track.outer_points)
        
        return is_off_track
    
    def _is_truncated(self) -> bool:
        """
        Check if the episode should be truncated (time limit).
        
        Note: Truncation is different from termination!
        - Termination = failure (agent did something wrong)
        - Truncation = time limit (episode just ran out of time)
        
        Returns:
            truncated: True if episode should end due to time limit
        """
        # TODO: Check if max steps reached
        return self.current_step >= self.max_episode_steps
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Questions:
        - Should the track be the same every time or randomized?
        - Should the car start at the same position or vary?
        - What information should be returned in the info dict?
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        # TODO: Reset step counter
        self.current_step = 0
        
        # TODO: Reset car to initial position
        # Where should the car start? (from config)
        p.resetBasePositionAndOrientation(self.car_id, self.config["spawn"]["position"], self.config["spawn"]["orientation"], physicsClientId=self.physics_client)
        p.resetBaseVelocity(self.car_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.physics_client)
        
        # TODO: Reset tracking variables
        self.last_position = self.config["spawn"]["position"]
        self.progress = 0.0
        
        # TODO: Optionally randomize track or starting position
        # How can you add variety to training?
        np.random.seed(seed)
        self.track = Track(self.config_path, seed)
        self.track.spawn_in_pybullet(self.physics_client)
        observation = self._get_observation()
        info = {"track_seed": seed, "starting_position": self.config["spawn"]["position"]}
        return observation, info
    
    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        This is the core function that connects actions to outcomes.
        
        Questions:
        - How do actions map to car controls?
        - How many physics steps should run per environment step?
        - What order should things happen? (action -> physics -> observation -> reward)
        
        Args:
            action: Action from the agent
            
        Returns:
            observation: New observation after step
            reward: Reward for this step
            terminated: Whether episode ended due to failure
            truncated: Whether episode ended due to time limit
            info: Additional information
        """
        # TODO: Increment step counter
        self.current_step += 1
        
        # TODO: Convert action to car controls
        # The TankDriveController expects forward_input and turn_input (both -1 to +1)
        # How do you map your action space to these inputs?
        forward_input = action[0]
        turn_input = action[1]
        
        # TODO: Apply controls to car
        # The controller.update() method expects keyboard keys, but we're using actions
        # How can you adapt the controller or directly set velocities?
        # Hint: You might need to modify how you use the controller, or set velocities directly
        p.resetBaseVelocity(self.car_id, [forward_input, 0, 0], [0, 0, turn_input], physicsClientId=self.physics_client)
        
        # TODO: Step physics simulation
        # How many physics steps should run? (config has time_step)
        for _ in range(int(1 / self.config["physics"]["time_step"])):
            p.stepSimulation(physicsClientId=self.physics_client)
        
        # TODO: Get new observation
        observation = self._get_observation()
        
        # TODO: Calculate reward
        reward = self._compute_reward(action)
        
        # TODO: Check termination and truncation
        terminated = self._is_terminated()
        truncated = self._is_truncated()
        
        # TODO: Prepare info dict
        # What information might be useful for debugging or logging?
        info = {"step": self.current_step, "reward": reward, "terminated": terminated, "truncated": truncated}
        
        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )
    
    def render(self):
        """
        Render the environment.
        
        Questions:
        - What should be rendered? (car, track, camera view?)
        - How do you render in PyBullet?
        - Should rendering affect performance?
        """
        if self.render_mode == "human":
            # TODO: PyBullet GUI rendering
            # How do you render in PyBullet GUI mode?
            # What camera settings might be useful?
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.physics_client)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=self.physics_client)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1, physicsClientId=self.physics_client)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1, physicsClientId=self.physics_client)
        elif self.render_mode == "rgb_array":
            # TODO: Return RGB array for video recording
            # How do you capture an image from PyBullet?
            # Hint: p.getCameraImage()
            return p.getCameraImage(self.config["camera"]["width"], self.config["camera"]["height"], self.config["camera"]["view_matrix"], self.config["camera"]["projection_matrix"], physicsClientId=self.physics_client)
    
    def close(self):
        """
        Clean up resources.
        
        Questions:
        - What needs to be cleaned up?
        - What happens if you don't close properly?
        """
        # TODO: Disconnect from PyBullet
        p.disconnect(physicsClientId=self.physics_client)

