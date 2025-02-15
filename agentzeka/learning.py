import torch
from torch.nn import Module, Linear
from typing import Dict, Any, List
import numpy as np
from .core import AgentModule

class RewardSystem:
	"""Manages rewards and learning signals for agents."""
	
	def __init__(self, discount_factor: float = 0.95):
		self.discount_factor = discount_factor
		self.reward_history = {}
		self.performance_metrics = {}
		
	def calculate_reward(self, agent_name: str, metrics: Dict[str, float]) -> float:
		reward = 0.0
		reward += metrics.get("task_completion", 0.0) * 1.0
		reward += metrics.get("efficiency", 0.0) * 0.5
		reward += metrics.get("innovation", 0.0) * 0.3
		
		if agent_name not in self.reward_history:
			self.reward_history[agent_name] = []
		self.reward_history[agent_name].append(reward)
		
		return reward

class AdaptiveAgent(AgentModule):
	"""Agent capable of learning and adapting."""
	
	def __init__(self, input_size: int, hidden_size: int = 64):
		super().__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.value_network = Linear(input_size, hidden_size)
		self.policy_network = Linear(hidden_size, input_size)
		self.reward_system = RewardSystem()
		self.state_history: List[Dict[str, Any]] = []
		
	def adapt(self, state: torch.Tensor, metrics: Dict[str, float]) -> None:
		reward = self.reward_system.calculate_reward(self._get_name(), metrics)
		hidden = torch.relu(self.value_network(state))
		action = torch.tanh(self.policy_network(hidden))
		loss = -torch.mean(action * reward + hidden.mean())
		loss.backward()
		
		self.state_history.append({
			"state": state.detach().numpy(),
			"metrics": metrics,
			"reward": reward,
			"loss": loss.item()
		})
		
	def _expert_forward(self, x: torch.Tensor, task: Dict[str, Any]) -> torch.Tensor:
		hidden = torch.relu(self.value_network(x))
		action = torch.tanh(self.policy_network(hidden))
		adapted_x = x + action
		
		metrics = {
			"task_completion": task.get("completion_rate", 0.0),
			"efficiency": task.get("efficiency", 0.0),
			"innovation": task.get("innovation", 0.0)
		}
		self.adapt(x, metrics)
		
		return adapted_x