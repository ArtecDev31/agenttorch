import torch
from torch.nn import Module, Linear
from typing import Dict, Any
from .core import AgentModule, AgentNetwork
from .experts import DataPreprocessingAgent, ModelTrainingAgent, HyperparameterAgent
from .learning import AdaptiveAgent

class AgentZekaLayer(Module):
	"""Integration layer between traditional PyTorch layers and AgentZeka system."""
	
	def __init__(self, in_features: int, out_features: int):
		super().__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.linear = Linear(in_features, out_features)
		self.agent_network = AgentNetwork()
		self._init_agents()
		
	def _init_agents(self):
		preprocessor = DataPreprocessingAgent(self.in_features)
		trainer = ModelTrainingAgent(self.out_features)
		hypertuner = HyperparameterAgent(self.out_features)
		adaptive = AdaptiveAgent(self.out_features)
		
		self.agent_network.add_agent("preprocessor", preprocessor)
		self.agent_network.add_agent("trainer", trainer)
		self.agent_network.add_agent("hypertuner", hypertuner)
		self.agent_network.add_agent("adaptive", adaptive)
		
		self.agent_network.connect_agents("preprocessor", "trainer")
		self.agent_network.connect_agents("trainer", "hypertuner")
		self.agent_network.connect_agents("hypertuner", "adaptive")
		
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		traditional_output = self.linear(x)
		agent_output = self.agent_network(x)
		return (traditional_output + agent_output) / 2

class AgentZekaModule(Module):
	"""Module for integrating AgentZeka into existing PyTorch models."""
	
	def __init__(self, model: Module):
		super().__init__()
		self.model = model
		self.agent_layers = torch.nn.ModuleList()
		self._convert_layers()
		
	def _convert_layers(self):
		for name, module in self.model.named_modules():
			if isinstance(module, Linear):
				agent_layer = AgentZekaLayer(module.in_features, module.out_features)
				agent_layer.linear.load_state_dict(module.state_dict())
				self.agent_layers.append(agent_layer)
				
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		output = x
		for layer in self.agent_layers:
			output = layer(output)
		return output

def convert_to_agentzeka(model: Module) -> AgentZekaModule:
	"""Convert a traditional PyTorch model to use AgentZeka."""
	return AgentZekaModule(model)