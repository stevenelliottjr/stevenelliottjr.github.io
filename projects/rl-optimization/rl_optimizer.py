import numpy as np
import tensorflow as tf
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os
import pickle
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from typing import Dict, List, Tuple, Union, Optional

class ResourceAllocationEnv(gym.Env):
    """
    A custom Gym environment for resource allocation problems.
    """
    
    def __init__(self, 
                 num_resources: int = 10, 
                 num_tasks: int = 5,
                 task_priorities: Optional[List[float]] = None,
                 constraints: Optional[Dict] = None,
                 reward_type: str = "weighted_completion",
                 max_steps: int = 100):
        """
        Initialize the resource allocation environment.
        
        Args:
            num_resources: Total number of resources to allocate
            num_tasks: Number of tasks that need resources
            task_priorities: Priority weights for each task (if None, will be randomly generated)
            constraints: Dictionary of constraints like {"max_per_task": 3}
            reward_type: Type of reward function to use
            max_steps: Maximum number of steps before the episode ends
        """
        super(ResourceAllocationEnv, self).__init__()
        
        self.num_resources = num_resources
        self.num_tasks = num_tasks
        
        # Set or generate task priorities
        if task_priorities is not None:
            if len(task_priorities) != num_tasks:
                raise ValueError(f"task_priorities length ({len(task_priorities)}) must match num_tasks ({num_tasks})")
            self.task_priorities = np.array(task_priorities)
        else:
            self.task_priorities = np.random.uniform(0.1, 1.0, size=num_tasks)
            # Normalize to sum to 1
            self.task_priorities = self.task_priorities / np.sum(self.task_priorities)
        
        # Set default constraints if not provided
        self.constraints = constraints or {"max_per_task": num_resources}
        self.max_per_task = self.constraints.get("max_per_task", num_resources)
        self.min_per_task = self.constraints.get("min_per_task", 0)
        
        self.reward_type = reward_type
        self.max_steps = max_steps
        
        # Action space: which task to allocate a resource to
        self.action_space = spaces.Discrete(num_tasks)
        
        # Observation space: current allocation state plus remaining resources
        # [task_1_allocation, task_2_allocation, ..., remaining_resources]
        self.observation_space = spaces.Box(
            low=0, 
            high=num_resources, 
            shape=(num_tasks + 1,), 
            dtype=np.int32
        )
        
        # Initialize state
        self.reset()
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.
        
        Returns:
            Observation of the initial state
        """
        super().reset(seed=seed)
        
        # Initial state: no resources allocated yet
        self.allocation = np.zeros(self.num_tasks, dtype=np.int32)
        self.remaining_resources = self.num_resources
        self.steps = 0
        
        # Create the observation
        observation = np.append(self.allocation, self.remaining_resources)
        
        # Empty info dict
        info = {}
        
        return observation, info
    
    def step(self, action):
        """
        Take a step in the environment by allocating a resource to a task.
        
        Args:
            action: Which task to allocate a resource to (0 to num_tasks-1)
            
        Returns:
            observation: The new state
            reward: The reward for this action
            terminated: Whether the episode is done
            truncated: Whether the episode was truncated
            info: Additional information
        """
        self.steps += 1
        
        # Check if the action is valid
        if action < 0 or action >= self.num_tasks:
            raise ValueError(f"Invalid action: {action}, must be between 0 and {self.num_tasks-1}")
        
        # Check constraints
        if self.allocation[action] >= self.max_per_task:
            reward = -10  # Penalty for violating max constraint
            terminated = False
            truncated = False
        elif self.remaining_resources <= 0:
            reward = 0  # No more resources to allocate
            terminated = True
            truncated = False
        else:
            # Valid action, allocate the resource
            self.allocation[action] += 1
            self.remaining_resources -= 1
            
            # Calculate reward based on the chosen reward type
            reward = self._calculate_reward()
            
            # Check if all resources are allocated
            if self.remaining_resources == 0:
                terminated = True
            else:
                terminated = False
                
            truncated = False
        
        # Check if we've reached the maximum number of steps
        if self.steps >= self.max_steps:
            truncated = True
        
        # Create the new observation
        observation = np.append(self.allocation, self.remaining_resources)
        
        # Additional info
        info = {
            "allocation": self.allocation.copy(),
            "remaining_resources": self.remaining_resources,
            "task_priorities": self.task_priorities
        }
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self):
        """
        Calculate the reward based on the current allocation.
        
        Different reward types incentivize different allocation strategies.
        
        Returns:
            float: The calculated reward
        """
        if self.reward_type == "weighted_completion":
            # Reward proportional to task priorities
            return np.sum(self.allocation * self.task_priorities)
        
        elif self.reward_type == "balanced":
            # Reward allocations that are proportional to priorities
            target_allocation = self.task_priorities * self.num_resources
            deviation = np.sum(np.abs(self.allocation - target_allocation))
            return -deviation
        
        elif self.reward_type == "sparse":
            # Only give reward when all resources are allocated
            if self.remaining_resources == 0:
                return np.sum(self.allocation * self.task_priorities) * 10
            else:
                return 0
        
        else:
            # Default to simple completion reward
            return np.sum(self.allocation) / self.num_resources
    
    def render(self):
        """
        Render the current state of the environment.
        
        Returns:
            None, but prints the current allocation
        """
        print(f"Step {self.steps}, Remaining resources: {self.remaining_resources}")
        for i in range(self.num_tasks):
            print(f"Task {i+1} (Priority: {self.task_priorities[i]:.2f}): " + 
                  "█" * self.allocation[i] + "░" * (self.max_per_task - self.allocation[i]))
        print("")
    
    def get_optimal_allocation(self):
        """
        Get the theoretically optimal allocation for this environment.
        
        This is a simplified version that just allocates resources proportionally 
        to task priorities, respecting constraints.
        
        Returns:
            numpy.ndarray: The optimal allocation
        """
        # Start with allocation proportional to priorities
        proportional = self.task_priorities * self.num_resources
        
        # Apply max constraint
        capped = np.minimum(proportional, self.max_per_task)
        
        # Apply min constraint
        with_min = np.maximum(capped, self.min_per_task)
        
        # Adjust to ensure total matches num_resources
        # This is a simplified approximation
        optimal = np.round(with_min).astype(np.int32)
        
        # Ensure we allocate exactly num_resources
        while np.sum(optimal) < self.num_resources:
            # Find the task with highest priority that isn't at max
            valid_tasks = np.where(optimal < self.max_per_task)[0]
            if len(valid_tasks) == 0:
                break
                
            best_task = valid_tasks[np.argmax(self.task_priorities[valid_tasks])]
            optimal[best_task] += 1
            
        while np.sum(optimal) > self.num_resources:
            # Find the task with lowest priority that isn't at min
            valid_tasks = np.where(optimal > self.min_per_task)[0]
            if len(valid_tasks) == 0:
                break
                
            worst_task = valid_tasks[np.argmin(self.task_priorities[valid_tasks])]
            optimal[worst_task] -= 1
            
        return optimal

class ResourceOptimizer:
    """
    A class that uses reinforcement learning to optimize resource allocation.
    """
    
    def __init__(self, 
                 resources: int = 10, 
                 tasks: int = 5,
                 task_priorities: Optional[List[float]] = None,
                 algorithm: str = "ppo",
                 constraints: Optional[Dict] = None,
                 reward_type: str = "weighted_completion"):
        """
        Initialize the resource optimizer.
        
        Args:
            resources: Number of resources to allocate
            tasks: Number of tasks requiring resources
            task_priorities: Priority weights for each task
            algorithm: RL algorithm to use ('ppo', 'dqn', 'a2c')
            constraints: Dictionary of allocation constraints
            reward_type: Type of reward function to use
        """
        self.resources = resources
        self.tasks = tasks
        self.task_priorities = task_priorities
        self.algorithm_name = algorithm.lower()
        self.constraints = constraints
        self.reward_type = reward_type
        
        # Create the environment
        self.env = ResourceAllocationEnv(
            num_resources=resources,
            num_tasks=tasks,
            task_priorities=task_priorities,
            constraints=constraints,
            reward_type=reward_type
        )
        
        # Wrap the environment with a monitor for logging
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        self.env = Monitor(self.env, log_dir)
        
        # Initialize the agent
        self._create_agent()
        
        # Tracking variables
        self.training_history = {
            "rewards": [],
            "episodes": []
        }
        self.trained = False
    
    def _create_agent(self):
        """
        Create the reinforcement learning agent based on the selected algorithm.
        """
        # Define policy kwargs based on environment
        policy_kwargs = dict(
            net_arch=[64, 64]  # Two hidden layers with 64 units each
        )
        
        if self.algorithm_name == "ppo":
            self.agent = PPO(
                "MlpPolicy",
                self.env,
                verbose=0,
                policy_kwargs=policy_kwargs
            )
        elif self.algorithm_name == "dqn":
            self.agent = DQN(
                "MlpPolicy",
                self.env,
                verbose=0,
                policy_kwargs=policy_kwargs
            )
        elif self.algorithm_name == "a2c":
            self.agent = A2C(
                "MlpPolicy",
                self.env,
                verbose=0,
                policy_kwargs=policy_kwargs
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm_name}. Use 'ppo', 'dqn', or 'a2c'.")
    
    def train(self, 
              episodes: int = 1000, 
              learning_rate: float = 0.0003, 
              gamma: float = 0.99,
              verbose: int = 0):
        """
        Train the RL agent to optimize resource allocation.
        
        Args:
            episodes: Number of training episodes
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor for future rewards
            verbose: Verbosity level (0: no output, 1: info, 2: debug)
            
        Returns:
            dict: Training history with rewards
        """
        # Set up a callback to evaluate the agent periodically
        eval_env = ResourceAllocationEnv(
            num_resources=self.resources,
            num_tasks=self.tasks,
            task_priorities=self.task_priorities,
            constraints=self.constraints,
            reward_type=self.reward_type
        )
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="./best_model",
            log_path="./logs",
            eval_freq=max(episodes // 10, 1),
            n_eval_episodes=10,
            verbose=0
        )
        
        # Set learning rate
        self.agent.learning_rate = learning_rate
        self.agent.gamma = gamma
        
        # Train the agent
        total_timesteps = episodes * self.env.max_steps
        
        if verbose > 0:
            print(f"Training {self.algorithm_name.upper()} agent for {episodes} episodes...")
        
        self.agent.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=verbose > 0
        )
        
        # Record training history
        self.training_history["episodes"] = list(range(episodes))
        
        # Evaluate the trained model
        mean_reward, std_reward = evaluate_policy(
            self.agent, 
            self.env, 
            n_eval_episodes=20
        )
        
        if verbose > 0:
            print(f"Mean reward after training: {mean_reward:.2f} ± {std_reward:.2f}")
        
        self.trained = True
        return self.training_history
    
    def allocate(self, 
                resources: Optional[int] = None, 
                task_priorities: Optional[List[float]] = None,
                constraints: Optional[Dict] = None):
        """
        Use the trained agent to allocate resources for a new scenario.
        
        Args:
            resources: Number of resources to allocate (if different from training)
            task_priorities: Task priorities (if different from training)
            constraints: Allocation constraints (if different from training)
            
        Returns:
            numpy.ndarray: The optimized allocation
        """
        if not self.trained:
            raise ValueError("Agent not trained. Call train() first.")
        
        # Create a new environment with the specified parameters
        resources = resources or self.resources
        task_priorities = task_priorities or self.task_priorities
        constraints = constraints or self.constraints
        
        env = ResourceAllocationEnv(
            num_resources=resources,
            num_tasks=self.tasks,
            task_priorities=task_priorities,
            constraints=constraints,
            reward_type=self.reward_type
        )
        
        # Run the agent to get an allocation
        observation, info = env.reset()
        done = False
        allocation = np.zeros(self.tasks, dtype=np.int32)
        
        while not done:
            action, _ = self.agent.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if done:
                allocation = info["allocation"]
        
        return allocation
    
    def compare_to_optimal(self, 
                          resources: Optional[int] = None, 
                          task_priorities: Optional[List[float]] = None,
                          constraints: Optional[Dict] = None):
        """
        Compare the RL agent's allocation to a theoretically optimal allocation.
        
        Args:
            resources: Number of resources to allocate
            task_priorities: Task priorities
            constraints: Allocation constraints
            
        Returns:
            dict: Comparison metrics and allocations
        """
        # Get the RL agent's allocation
        rl_allocation = self.allocate(resources, task_priorities, constraints)
        
        # Create an environment for the optimal allocation
        resources = resources or self.resources
        task_priorities = task_priorities or self.task_priorities
        constraints = constraints or self.constraints
        
        env = ResourceAllocationEnv(
            num_resources=resources,
            num_tasks=self.tasks,
            task_priorities=task_priorities,
            constraints=constraints,
            reward_type=self.reward_type
        )
        
        # Get the theoretical optimal allocation
        optimal_allocation = env.get_optimal_allocation()
        
        # Calculate weighted values (higher is better)
        rl_weighted_value = np.sum(rl_allocation * env.task_priorities)
        optimal_weighted_value = np.sum(optimal_allocation * env.task_priorities)
        
        # Calculate optimality percentage
        if optimal_weighted_value > 0:
            optimality_percentage = (rl_weighted_value / optimal_weighted_value) * 100
        else:
            optimality_percentage = 100.0
        
        return {
            "rl_allocation": rl_allocation,
            "optimal_allocation": optimal_allocation,
            "rl_weighted_value": rl_weighted_value,
            "optimal_weighted_value": optimal_weighted_value,
            "optimality_percentage": optimality_percentage,
            "task_priorities": env.task_priorities
        }
    
    def visualize_allocation(self, 
                            allocation: Optional[np.ndarray] = None,
                            task_priorities: Optional[np.ndarray] = None,
                            comparison: Optional[Dict] = None,
                            title: str = "Resource Allocation"):
        """
        Visualize the resource allocation.
        
        Args:
            allocation: Resource allocation to visualize (if None, will use the latest)
            task_priorities: Task priorities (if None, will use the environment's)
            comparison: Comparison dict from compare_to_optimal() (if provided, will show comparison)
            title: Plot title
            
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        if comparison is not None:
            return self._visualize_comparison(comparison, title)
        
        if allocation is None:
            allocation = self.allocate()
        
        task_priorities = task_priorities or self.env.task_priorities
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Task names
        task_names = [f"Task {i+1}" for i in range(len(allocation))]
        
        # Create a DataFrame for the visualization
        data = pd.DataFrame({
            "Task": task_names,
            "Resources": allocation,
            "Priority": task_priorities
        })
        
        # Sort by priority
        data = data.sort_values("Priority", ascending=False)
        
        # Create the bar chart
        bars = sns.barplot(x="Resources", y="Task", data=data, ax=ax, 
                          palette="viridis", orient="h")
        
        # Add priority labels
        for i, (_, row) in enumerate(data.iterrows()):
            ax.text(row["Resources"] + 0.1, i, f"Priority: {row['Priority']:.2f}", 
                   va="center")
        
        ax.set_title(title)
        ax.set_xlabel("Resources Allocated")
        ax.set_ylabel("Task")
        
        plt.tight_layout()
        return fig
    
    def _visualize_comparison(self, comparison: Dict, title: str):
        """
        Visualize a comparison between RL and optimal allocations.
        
        Args:
            comparison: Comparison dict from compare_to_optimal()
            title: Plot title
            
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        # Extract data from comparison
        rl_allocation = comparison["rl_allocation"]
        optimal_allocation = comparison["optimal_allocation"]
        task_priorities = comparison["task_priorities"]
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Task names
        task_names = [f"Task {i+1}" for i in range(len(rl_allocation))]
        
        # Prepare data for plotting
        data = pd.DataFrame({
            "Task": np.repeat(task_names, 2),
            "Allocation": np.concatenate([rl_allocation, optimal_allocation]),
            "Method": ["RL Agent"] * len(rl_allocation) + ["Optimal"] * len(optimal_allocation),
            "Priority": np.tile(task_priorities, 2)
        })
        
        # Sort by priority
        task_order = pd.DataFrame({"Task": task_names, "Priority": task_priorities})
        task_order = task_order.sort_values("Priority", ascending=True)
        
        # Create the grouped bar chart
        sns.barplot(x="Allocation", y="Task", hue="Method", data=data, 
                  palette=["#2C7FB8", "#7FBC41"], orient="h", order=task_order["Task"])
        
        # Add priority labels
        for i, task in enumerate(task_order["Task"]):
            task_idx = int(task.split(" ")[1]) - 1
            ax.text(-0.5, i, f"Priority: {task_priorities[task_idx]:.2f}", va="center")
        
        # Add performance metrics
        optimality = comparison["optimality_percentage"]
        ax.text(0.5, 1.05, f"RL Performance: {optimality:.1f}% of Optimal", 
               transform=ax.transAxes, fontsize=12, fontweight="bold")
        
        ax.set_title(title)
        ax.set_xlabel("Resources Allocated")
        ax.set_ylabel("Task (sorted by priority)")
        
        plt.tight_layout()
        return fig
    
    def plot_training_history(self):
        """
        Plot the training history of the agent.
        
        Returns:
            matplotlib.figure.Figure: The training history plot
        """
        if not self.trained:
            raise ValueError("No training history available. Train the agent first.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Read reward data from the monitor
        monitor_data = pd.read_csv(os.path.join("logs", "monitor.csv"), skiprows=1)
        
        # Plot rewards over episodes
        sns.lineplot(x=range(len(monitor_data)), y="r", data=monitor_data, ax=ax)
        
        ax.set_title(f"Training Progress - {self.algorithm_name.upper()}")
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Reward")
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def save(self, path: str):
        """
        Save the trained agent and optimizer configuration.
        
        Args:
            path: Path where to save the agent and configuration
            
        Returns:
            None
        """
        if not self.trained:
            raise ValueError("Agent not trained. Call train() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save the agent
        self.agent.save(os.path.join(path, "agent"))
        
        # Save the configuration
        config = {
            "resources": self.resources,
            "tasks": self.tasks,
            "task_priorities": self.task_priorities.tolist() if self.task_priorities is not None else None,
            "algorithm": self.algorithm_name,
            "constraints": self.constraints,
            "reward_type": self.reward_type
        }
        
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)
    
    @classmethod
    def load(cls, path: str):
        """
        Load a trained agent and optimizer configuration.
        
        Args:
            path: Path from where to load the agent and configuration
            
        Returns:
            ResourceOptimizer: Loaded optimizer with trained agent
        """
        # Load the configuration
        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)
        
        # Create a new optimizer
        optimizer = cls(
            resources=config["resources"],
            tasks=config["tasks"],
            task_priorities=config["task_priorities"],
            algorithm=config["algorithm"],
            constraints=config["constraints"],
            reward_type=config["reward_type"]
        )
        
        # Load the agent
        if config["algorithm"] == "ppo":
            optimizer.agent = PPO.load(os.path.join(path, "agent"), optimizer.env)
        elif config["algorithm"] == "dqn":
            optimizer.agent = DQN.load(os.path.join(path, "agent"), optimizer.env)
        elif config["algorithm"] == "a2c":
            optimizer.agent = A2C.load(os.path.join(path, "agent"), optimizer.env)
        
        optimizer.trained = True
        return optimizer