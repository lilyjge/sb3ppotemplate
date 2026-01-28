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
        gravity = self.config["physics"]["gravity"]
        p.setGravity(0, 0, gravity, physicsClientId=self.physics_client)
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
        # Spawn on track centerline (first point)
        spawn_pos = self.track.centerline[0].copy()
        spawn_orn = self.config["spawn"]["orientation"]
        p.resetBasePositionAndOrientation(self.car_id, spawn_pos, spawn_orn, physicsClientId=self.physics_client)
        # Reset velocity to zero
        p.resetBaseVelocity(self.car_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.physics_client)
        
        # Initialize controller
        # TODO: Create the TankDriveController. What does it need?
        self.controller = TankDriveController(self.config_path, self.car_id, self.physics_client)
        # Update fixed height to match centerline height
        self.controller.fixed_height = self.track.centerline[0][2]
        
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
        self.last_position = self.track.centerline[0].copy()
        self.progress = 0.0
        self.episode_reward = 0.0
        # Track recent positions for stuck detection
        self.position_history = [self.last_position.copy()]
        self.stuck_threshold = 0.01  # Minimum movement per step to not be considered stuck
        self.stuck_steps = 0
        self.stuck_max_steps = 50  # Terminate if stuck for this many steps
        # Track progress along centerline (which segment index we're closest to)
        self.last_centerline_index = 0
        self.total_lap_progress = 0.0  # Track cumulative progress around track
        
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
        # Use the same logic as termination check for consistency
        car_pos_2d = np.array(car_pos[:2])
        distances_to_inner = np.linalg.norm(self.track.inner_points[:, :2] - car_pos_2d, axis=1)
        distances_to_outer = np.linalg.norm(self.track.outer_points[:, :2] - car_pos_2d, axis=1)
        min_dist_to_inner = np.min(distances_to_inner)
        min_dist_to_outer = np.min(distances_to_outer)
        inner_margin = 0.05
        outer_margin = 0.15
        is_on_track = not ((min_dist_to_inner < inner_margin) or (min_dist_to_outer > self.track.track_width + outer_margin))
        
        # TODO: Calculate progress
        # How do you measure progress around the track?
        # Think about: Distance traveled, laps completed, distance from start, etc.
        # Measure forward progress along the track centerline, not just Euclidean distance
        car_pos_2d = np.array(car_pos[:2])
        centerline_2d = self.track.centerline[:, :2]
        num_segments = len(centerline_2d)
        
        # Find closest point on centerline to current position
        distances_to_centerline = np.linalg.norm(centerline_2d - car_pos_2d, axis=1)
        current_centerline_idx = np.argmin(distances_to_centerline)
        
        # Find closest point on centerline to last position
        last_pos_2d = np.array(self.last_position[:2])
        distances_to_centerline_last = np.linalg.norm(centerline_2d - last_pos_2d, axis=1)
        last_centerline_idx = np.argmin(distances_to_centerline_last)
        
        # Calculate forward progress along centerline
        # Define "forward" as increasing index (track direction), "backward" as decreasing index
        # The centerline points are ordered, so forward = increasing index (with wrap)
        forward_steps = (current_centerline_idx - last_centerline_idx) % num_segments
        backward_steps = (last_centerline_idx - current_centerline_idx) % num_segments
        
        # Get track tangent direction at current position (forward direction along track)
        next_idx = (current_centerline_idx + 1) % num_segments
        track_forward_dir = centerline_2d[next_idx] - centerline_2d[current_centerline_idx]
        track_forward_dir = track_forward_dir / (np.linalg.norm(track_forward_dir) + 1e-6)  # Normalize
        
        # Get car's velocity direction (2D) to determine actual movement direction
        car_vel_2d = np.array(car_vel[:2])
        car_speed_2d = np.linalg.norm(car_vel_2d)
        velocity_alignment = 0.0
        if car_speed_2d > 0.01:  # Only use velocity if car is actually moving
            car_vel_dir = car_vel_2d / car_speed_2d
            # Dot product: positive = moving forward along track, negative = moving backward
            velocity_alignment = np.dot(car_vel_dir, track_forward_dir)
        
        # Determine direction using velocity as primary signal (more reliable than segment jumps)
        # If velocity alignment is positive, we're moving forward; if negative, backward
        # Use segment progress as secondary signal when velocity is ambiguous
        if abs(velocity_alignment) > 0.1:  # Clear velocity signal
            is_moving_forward = velocity_alignment > 0.0
            # Use segment progress to determine magnitude, but direction from velocity
            if is_moving_forward:
                segment_progress = forward_steps
            else:
                segment_progress = backward_steps
        else:  # Velocity ambiguous (slow or perpendicular), use segment progress
            # Prefer forward direction (increasing index) unless clearly going backward
            if forward_steps <= backward_steps:
                is_moving_forward = True
                segment_progress = forward_steps
            else:
                is_moving_forward = False
                segment_progress = backward_steps
        
        # Calculate reward based on direction
        if is_moving_forward:
            # Moving forward - reward progress
            progress_reward = segment_progress / num_segments
            # Boost reward if velocity is well-aligned with track direction
            if velocity_alignment > 0.7:
                progress_reward *= 1.5  # Bonus for excellent alignment
            # Also add reward for staying close to centerline
            distance_to_centerline = distances_to_centerline[current_centerline_idx]
            centerline_alignment_bonus = max(0, 0.5 * (1.0 - distance_to_centerline / self.track.track_width))
        else:
            # Moving backward - penalize heavily
            progress_reward = -2.0 * segment_progress / num_segments  # Heavy penalty for going backward
            centerline_alignment_bonus = 0
        
        # Update tracking
        self.last_centerline_index = current_centerline_idx
        self.total_lap_progress += progress_reward
        
        # TODO: Calculate speed reward
        # Should the agent be rewarded for speed? Why or why not?
        # Only reward forward speed (velocity in forward direction)
        forward_speed = np.linalg.norm(car_vel[:2])  # 2D speed
        speed_reward = forward_speed if progress_reward > 0 else 0  # Only reward if making forward progress
        
        # TODO: Calculate penalty for going off track
        # How severe should the penalty be? Should it end the episode?
        off_track_penalty = -10.0 if not is_on_track else 0
        
        # TODO: Combine rewards
        # How do you weight different reward components?
        # What happens if rewards are too large or too small?
        # Reward forward progress heavily, add speed bonus, penalize off-track
        reward = 50.0 * progress_reward + 5.0 * speed_reward + 2.0 * centerline_alignment_bonus + off_track_penalty
        
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
        car_pos_array = np.array(car_pos)
        
        # Check if car is off track
        # Compute distances to inner and outer boundaries (using 2D projection, ignoring Z)
        car_pos_2d = car_pos_array[:2]
        distances_to_inner = np.linalg.norm(self.track.inner_points[:, :2] - car_pos_2d, axis=1)
        distances_to_outer = np.linalg.norm(self.track.outer_points[:, :2] - car_pos_2d, axis=1)
        min_dist_to_inner = np.min(distances_to_inner)
        min_dist_to_outer = np.min(distances_to_outer)
        
        # Car is off-track if:
        # 1. It's inside the inner boundary (distance to inner is very small, meaning it's past the inner boundary)
        # 2. It's outside the outer boundary (distance to outer is larger than track width, meaning it's past the outer boundary)
        # Use a margin to account for car size and track boundaries
        inner_margin = 0.05  # Car is off-track if within 5cm of inner boundary (inside it)
        outer_margin = 0.15  # Car is off-track if more than track_width + margin from outer boundary
        is_off_track = (min_dist_to_inner < inner_margin) or (min_dist_to_outer > self.track.track_width + outer_margin)
        
        # Check if car is stuck (hasn't moved much recently)
        if len(self.position_history) >= self.stuck_max_steps:
            # Check total distance moved in last N steps
            recent_positions = np.array(self.position_history[-self.stuck_max_steps:])
            total_distance = np.sum(np.linalg.norm(np.diff(recent_positions[:, :2], axis=0), axis=1))
            is_stuck = total_distance < self.stuck_threshold * self.stuck_max_steps
        else:
            is_stuck = False
        
        return is_off_track or is_stuck
    
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
        
        # TODO: Optionally randomize track or starting position
        # How can you add variety to training?
        # Regenerate track first (if seed is provided), then position car on new track
        if seed is not None:
            np.random.seed(seed)
            self.track = Track(self.config_path, seed)
            self.track.spawn_in_pybullet(self.physics_client)
            # Update fixed height to match new track centerline
            self.controller.fixed_height = self.track.centerline[0][2]
        
        # TODO: Reset car to initial position
        # Where should the car start? (from config)
        # Spawn on track centerline (first point)
        spawn_pos = self.track.centerline[0].copy()
        spawn_orn = self.config["spawn"]["orientation"]
        p.resetBasePositionAndOrientation(self.car_id, spawn_pos, spawn_orn, physicsClientId=self.physics_client)
        p.resetBaseVelocity(self.car_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.physics_client)
        
        # TODO: Reset tracking variables
        self.last_position = spawn_pos.copy()
        self.progress = 0.0
        # Reset stuck detection
        self.position_history = [spawn_pos.copy()]
        self.stuck_steps = 0
        # Reset centerline progress tracking
        self.last_centerline_index = 0
        self.total_lap_progress = 0.0
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
        forward_input = np.clip(action[0], -1.0, 1.0)
        turn_input = np.clip(action[1], -1.0, 1.0)
        
        # TODO: Apply controls to car
        # The controller.update() method expects keyboard keys, but we're using actions
        # How can you adapt the controller or directly set velocities?
        # Hint: You might need to modify how you use the controller, or set velocities directly
        # Scale to actual velocities
        linear_velocity = forward_input * self.controller.max_linear_velocity
        angular_velocity = turn_input * self.controller.max_angular_velocity
        
        # Get car orientation to compute forward direction
        car_pos, car_orn = p.getBasePositionAndOrientation(self.car_id, physicsClientId=self.physics_client)
        
        # Lock Z position to fixed height
        if abs(car_pos[2] - self.controller.fixed_height) > 0.001:
            p.resetBasePositionAndOrientation(
                self.car_id,
                [car_pos[0], car_pos[1], self.controller.fixed_height],
                car_orn,
                physicsClientId=self.physics_client
            )
            car_pos, car_orn = p.getBasePositionAndOrientation(self.car_id, physicsClientId=self.physics_client)
        
        # Compute forward direction in world frame
        rot_matrix = np.array(p.getMatrixFromQuaternion(car_orn)).reshape((3, 3))
        car_forward = rot_matrix[:, 0]  # X-axis in car's local frame
        
        linear_velocity_world = linear_velocity * car_forward
        
        # Set velocity (Z component locked to 0)
        p.resetBaseVelocity(
            objectUniqueId=self.car_id,
            linearVelocity=[linear_velocity_world[0], linear_velocity_world[1], 0.0],
            angularVelocity=[0.0, 0.0, angular_velocity],
            physicsClientId=self.physics_client
        )
        
        # TODO: Step physics simulation
        # How many physics steps should run? (config has time_step)
        for _ in range(int(1 / self.config["physics"]["time_step"])):
            p.stepSimulation(physicsClientId=self.physics_client)
        
        # TODO: Get new observation
        observation = self._get_observation()
        
        # Update position history for stuck detection
        car_pos, _ = p.getBasePositionAndOrientation(self.car_id, physicsClientId=self.physics_client)
        self.position_history.append(np.array(car_pos))
        # Keep only recent history (last stuck_max_steps + some buffer)
        if len(self.position_history) > self.stuck_max_steps + 10:
            self.position_history.pop(0)
        
        # TODO: Calculate reward
        reward = self._compute_reward(action)
        
        # Update last position for progress tracking
        self.last_position = np.array(car_pos)
        
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

