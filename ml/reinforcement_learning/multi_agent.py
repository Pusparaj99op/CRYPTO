"""
Multi-agent reinforcement learning implementation for cryptocurrency trading.

This module provides implementations of multi-agent systems that can be
used to simulate complex market dynamics or create collaborative trading systems.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Union, Optional, Any
import os
import pandas as pd
from collections import deque

from .dqn import DQNAgent, DQNTrader
from .ppo import PPOAgent, PPOTrader
from .a2c import A2CAgent, A2CTrader
from .ddpg import DDPGAgent, DDPGTrader
from .sac import SACAgent, SACTrader

class MultiAgentSystem:
    """
    Base class for multi-agent reinforcement learning systems.
    
    This provides the foundation for implementing various multi-agent approaches
    in cryptocurrency trading environments.
    """
    
    def __init__(
        self,
        num_agents: int,
        agent_type: str = 'dqn',
        shared_state: bool = False,
        agent_params: Optional[List[Dict]] = None,
        communication_enabled: bool = False
    ):
        """
        Initialize multi-agent system.
        
        Args:
            num_agents: Number of agents in the system
            agent_type: Type of agent ('dqn', 'ppo', 'a2c', 'ddpg', 'sac')
            shared_state: Whether agents share the same state observations
            agent_params: List of dictionaries with parameters for each agent
            communication_enabled: Whether agents can communicate
        """
        self.num_agents = num_agents
        self.agent_type = agent_type.lower()
        self.shared_state = shared_state
        self.communication_enabled = communication_enabled
        
        # Create agent instances
        self.agents = []
        
        if agent_params is None:
            agent_params = [{}] * num_agents
            
        for i in range(num_agents):
            params = agent_params[i] if i < len(agent_params) else {}
            self.agents.append(self._create_agent(params))
            
        # Communication buffer (if enabled)
        if communication_enabled:
            self.message_buffer = [[] for _ in range(num_agents)]
            
        # Metrics tracking
        self.rewards = np.zeros((num_agents,))
        self.cumulative_rewards = np.zeros((num_agents,))
        self.episode_history = []
        
    def _create_agent(self, params: Dict) -> Any:
        """
        Create an individual agent based on the specified agent type.
        
        Args:
            params: Parameters for agent initialization
            
        Returns:
            Agent instance
        """
        if self.agent_type == 'dqn':
            return DQNTrader(**params)
        elif self.agent_type == 'ppo':
            return PPOTrader(**params)
        elif self.agent_type == 'a2c':
            return A2CTrader(**params)
        elif self.agent_type == 'ddpg':
            return DDPGTrader(**params)
        elif self.agent_type == 'sac':
            return SACTrader(**params)
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")
            
    def reset(self) -> None:
        """Reset all agents and system state."""
        for agent in self.agents:
            agent.reset()
            
        self.rewards = np.zeros((self.num_agents,))
        self.cumulative_rewards = np.zeros((self.num_agents,))
        
        if self.communication_enabled:
            self.message_buffer = [[] for _ in range(self.num_agents)]
            
    def get_actions(self, states: List[np.ndarray], deterministic: bool = False) -> List:
        """
        Get actions from all agents based on their states.
        
        Args:
            states: List of state observations for each agent
            deterministic: Whether to use deterministic action selection
            
        Returns:
            List of actions from all agents
        """
        actions = []
        
        for i, agent in enumerate(self.agents):
            # If agents share state, use the first state for all
            state = states[0] if self.shared_state else states[i]
            
            # If communication is enabled, augment state with messages
            if self.communication_enabled and len(self.message_buffer[i]) > 0:
                messages = np.mean(self.message_buffer[i], axis=0)
                if isinstance(state, np.ndarray):
                    augmented_state = np.concatenate([state, messages])
                else:
                    augmented_state = np.concatenate([np.array(state), messages])
                action = agent.act(augmented_state, deterministic=deterministic)
            else:
                action = agent.act(state, deterministic=deterministic)
                
            actions.append(action)
            
        return actions
        
    def update(self, states: List[np.ndarray], actions: List, rewards: List[float], 
              next_states: List[np.ndarray], dones: List[bool]) -> None:
        """
        Update all agents with their experiences.
        
        Args:
            states: Current states for each agent
            actions: Actions taken by each agent
            rewards: Rewards received by each agent
            next_states: Next states for each agent
            dones: Done flags for each agent
        """
        for i, agent in enumerate(self.agents):
            # Record experience for agent
            state = states[0] if self.shared_state else states[i]
            next_state = next_states[0] if self.shared_state else next_states[i]
            
            agent.remember(state, actions[i], rewards[i], next_state, dones[i])
            
            # Update rewards tracking
            self.rewards[i] = rewards[i]
            self.cumulative_rewards[i] += rewards[i]
            
        # Store episode history if any agent is done
        if any(dones):
            self.episode_history.append(self.cumulative_rewards.copy())
            
    def train(self) -> List[Dict]:
        """
        Train all agents.
        
        Returns:
            List of training metrics for each agent
        """
        metrics = []
        
        for agent in self.agents:
            result = agent.train()
            metrics.append(result)
            
        return metrics
        
    def communicate(self, messages: List[np.ndarray], agent_ids: List[int]) -> None:
        """
        Enable communication between agents.
        
        Args:
            messages: List of message vectors from source agents
            agent_ids: IDs of source agents
        """
        if not self.communication_enabled:
            return
            
        for i, message in enumerate(messages):
            source_id = agent_ids[i]
            
            # Add message to all other agents' message buffers
            for j in range(self.num_agents):
                if j != source_id:
                    self.message_buffer[j].append(message)
                    
    def save(self, directory: str) -> None:
        """
        Save all agents to disk.
        
        Args:
            directory: Directory to save agents in
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        for i, agent in enumerate(self.agents):
            agent_dir = os.path.join(directory, f"agent_{i}")
            os.makedirs(agent_dir, exist_ok=True)
            agent.save(agent_dir)
            
        # Save system parameters
        np.save(os.path.join(directory, "cumulative_rewards.npy"), self.cumulative_rewards)
        np.save(os.path.join(directory, "episode_history.npy"), self.episode_history)
        
        # Save configuration
        config = {
            "num_agents": self.num_agents,
            "agent_type": self.agent_type,
            "shared_state": self.shared_state,
            "communication_enabled": self.communication_enabled
        }
        
        with open(os.path.join(directory, "config.txt"), "w") as f:
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
                
    def load(self, directory: str) -> None:
        """
        Load all agents from disk.
        
        Args:
            directory: Directory to load agents from
        """
        for i, agent in enumerate(self.agents):
            agent_dir = os.path.join(directory, f"agent_{i}")
            if os.path.exists(agent_dir):
                agent.load(agent_dir)
                
        # Load system parameters
        if os.path.exists(os.path.join(directory, "cumulative_rewards.npy")):
            self.cumulative_rewards = np.load(os.path.join(directory, "cumulative_rewards.npy"))
            
        if os.path.exists(os.path.join(directory, "episode_history.npy")):
            self.episode_history = np.load(os.path.join(directory, "episode_history.npy"))
            
    def plot_performance(self, save_path: Optional[str] = None) -> None:
        """
        Plot performance metrics for all agents.
        
        Args:
            save_path: Path to save the plot image
        """
        if not self.episode_history:
            print("No episode history to plot.")
            return
            
        episode_history = np.array(self.episode_history)
        
        plt.figure(figsize=(12, 8))
        
        # Plot cumulative rewards for each agent
        for i in range(self.num_agents):
            plt.plot(episode_history[:, i], label=f"Agent {i}")
            
        plt.title("Cumulative Rewards per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Reward")
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()

class CompetitiveAgents(MultiAgentSystem):
    """
    Multi-agent system where agents compete with each other.
    
    This class implements a competitive multi-agent framework where each agent
    maximizes its own reward, potentially at the expense of others.
    """
    
    def __init__(
        self,
        num_agents: int,
        agent_type: str = 'dqn',
        zero_sum: bool = False,
        **kwargs
    ):
        """
        Initialize competitive multi-agent system.
        
        Args:
            num_agents: Number of agents in the system
            agent_type: Type of agent ('dqn', 'ppo', 'a2c', 'ddpg', 'sac')
            zero_sum: Whether the environment is zero-sum
            **kwargs: Additional arguments for MultiAgentSystem
        """
        super().__init__(num_agents, agent_type, **kwargs)
        self.zero_sum = zero_sum
        
    def update(self, states: List[np.ndarray], actions: List, rewards: List[float], 
              next_states: List[np.ndarray], dones: List[bool]) -> None:
        """
        Update agents with competitive reward shaping.
        
        Args:
            states: Current states for each agent
            actions: Actions taken by each agent
            rewards: Rewards received by each agent
            next_states: Next states for each agent
            dones: Done flags for each agent
        """
        if self.zero_sum:
            # In zero-sum games, the sum of all rewards must be zero
            reward_sum = sum(rewards)
            adjusted_rewards = [r - reward_sum / self.num_agents for r in rewards]
        else:
            # In competitive (but not zero-sum) games, shape rewards based on relative performance
            max_reward = max(rewards)
            min_reward = min(rewards)
            range_reward = max_reward - min_reward if max_reward > min_reward else 1.0
            
            # Normalize rewards to penalize underperforming agents
            adjusted_rewards = [(r - min_reward) / range_reward for r in rewards]
            
        # Update with adjusted rewards
        super().update(states, actions, adjusted_rewards, next_states, dones)
        
    def evaluate_competition(self) -> Dict:
        """
        Evaluate competitive metrics.
        
        Returns:
            Dictionary of competitive metrics
        """
        if not self.episode_history:
            return {}
            
        episode_history = np.array(self.episode_history)
        
        # Calculate win rates (who got the highest reward in each episode)
        wins = np.zeros(self.num_agents)
        for episode in episode_history:
            winner = np.argmax(episode)
            wins[winner] += 1
            
        win_rates = wins / len(episode_history) if len(episode_history) > 0 else wins
        
        # Calculate average rewards
        avg_rewards = np.mean(episode_history, axis=0)
        
        # Calculate dominance (how much an agent outperforms others)
        dominance = np.zeros(self.num_agents)
        for i in range(self.num_agents):
            others = np.concatenate([episode_history[:, :i], episode_history[:, i+1:]], axis=1)
            avg_others = np.mean(others, axis=1)
            dominance[i] = np.mean(episode_history[:, i] - avg_others)
            
        return {
            "win_rates": win_rates,
            "avg_rewards": avg_rewards,
            "dominance": dominance
        }

class CooperativeAgents(MultiAgentSystem):
    """
    Multi-agent system where agents cooperate toward a common goal.
    
    This class implements a cooperative multi-agent framework where agents work
    together to maximize a shared reward.
    """
    
    def __init__(
        self,
        num_agents: int,
        agent_type: str = 'dqn',
        shared_reward: bool = True,
        credit_assignment: bool = False,
        **kwargs
    ):
        """
        Initialize cooperative multi-agent system.
        
        Args:
            num_agents: Number of agents in the system
            agent_type: Type of agent ('dqn', 'ppo', 'a2c', 'ddpg', 'sac')
            shared_reward: Whether all agents receive the same reward
            credit_assignment: Whether to perform credit assignment
            **kwargs: Additional arguments for MultiAgentSystem
        """
        super().__init__(num_agents, agent_type, **kwargs)
        self.shared_reward = shared_reward
        self.credit_assignment = credit_assignment
        
        # History of actions for credit assignment
        if credit_assignment:
            self.action_history = deque(maxlen=10)
            
    def update(self, states: List[np.ndarray], actions: List, rewards: List[float], 
              next_states: List[np.ndarray], dones: List[bool]) -> None:
        """
        Update agents with cooperative reward shaping.
        
        Args:
            states: Current states for each agent
            actions: Actions taken by each agent
            rewards: Rewards received by each agent
            next_states: Next states for each agent
            dones: Done flags for each agent
        """
        if self.credit_assignment:
            # Store actions for credit assignment
            self.action_history.append(actions)
            
            # Calculate correlation between agent actions and total reward
            adjusted_rewards = rewards.copy()
            
            if len(self.action_history) >= 5:  # Need some history for correlation
                total_rewards = sum(rewards)
                
                # Simple credit assignment based on action consistency
                action_consistency = []
                
                for i in range(self.num_agents):
                    # Check how consistent the agent's actions have been
                    agent_actions = [ah[i] for ah in self.action_history]
                    if isinstance(agent_actions[0], (np.ndarray, list)):
                        # For continuous actions, use variance as a measure of consistency
                        consistency = 1.0 / (1.0 + np.var(np.array(agent_actions)))
                    else:
                        # For discrete actions, use mode frequency
                        unique, counts = np.unique(agent_actions, return_counts=True)
                        consistency = np.max(counts) / len(agent_actions)
                        
                    action_consistency.append(consistency)
                    
                # Normalize consistency scores
                total_consistency = sum(action_consistency)
                if total_consistency > 0:
                    normalized_consistency = [c / total_consistency for c in action_consistency]
                    
                    # Adjust rewards based on action consistency
                    for i in range(self.num_agents):
                        adjusted_rewards[i] = total_rewards * normalized_consistency[i]
        
        elif self.shared_reward:
            # In fully cooperative scenarios, all agents receive the team's total reward
            team_reward = sum(rewards)
            adjusted_rewards = [team_reward] * self.num_agents
        else:
            adjusted_rewards = rewards
            
        # Update with adjusted rewards
        super().update(states, actions, adjusted_rewards, next_states, dones)
        
    def evaluate_cooperation(self) -> Dict:
        """
        Evaluate cooperative metrics.
        
        Returns:
            Dictionary of cooperative metrics
        """
        if not self.episode_history:
            return {}
            
        episode_history = np.array(self.episode_history)
        
        # Calculate team performance (sum of all agent rewards)
        team_rewards = np.sum(episode_history, axis=1)
        
        # Calculate average team reward
        avg_team_reward = np.mean(team_rewards)
        
        # Calculate fairness (how evenly rewards are distributed)
        reward_std = np.std(episode_history, axis=1)
        fairness = 1.0 - np.mean(reward_std) / (np.mean(team_rewards) + 1e-10)
        
        # Calculate contribution (correlation between agent rewards and team reward)
        contribution = np.zeros(self.num_agents)
        for i in range(self.num_agents):
            agent_rewards = episode_history[:, i]
            # Correlation coefficient between agent and team rewards
            contribution[i] = np.corrcoef(agent_rewards, team_rewards)[0, 1]
            
        return {
            "avg_team_reward": avg_team_reward,
            "fairness": fairness,
            "contribution": contribution,
            "team_reward_trend": team_rewards
        }
        
    def plot_cooperation_metrics(self, save_path: Optional[str] = None) -> None:
        """
        Plot cooperation metrics.
        
        Args:
            save_path: Path to save the plot image
        """
        metrics = self.evaluate_cooperation()
        if not metrics:
            print("No metrics available to plot.")
            return
            
        plt.figure(figsize=(15, 10))
        
        # Plot team reward trend
        plt.subplot(2, 2, 1)
        plt.plot(metrics["team_reward_trend"])
        plt.title("Team Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Total Team Reward")
        plt.grid(True)
        
        # Plot agent contribution
        plt.subplot(2, 2, 2)
        plt.bar(range(self.num_agents), metrics["contribution"])
        plt.title("Agent Contribution to Team Success")
        plt.xlabel("Agent")
        plt.ylabel("Contribution (Correlation)")
        plt.grid(True)
        
        # Plot fairness
        plt.subplot(2, 2, 3)
        plt.bar(["Fairness"], [metrics["fairness"]])
        plt.title("Reward Distribution Fairness")
        plt.ylim(0, 1)
        plt.grid(True)
        
        # Add summary text
        plt.figtext(0.5, 0.01, f"Average Team Reward: {metrics['avg_team_reward']:.2f}\n"
                           f"Fairness Index: {metrics['fairness']:.2f}",
                  ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show() 