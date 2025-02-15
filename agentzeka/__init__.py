"""
AgentZeka: Agent-based neural network system for PyTorch
"""

from .core import AgentModule, AgentNetwork
from .experts import DataPreprocessingAgent, ModelTrainingAgent, HyperparameterAgent
from .learning import AdaptiveAgent, RewardSystem
from .integration import AgentZekaLayer, AgentZekaModule, convert_to_agentzeka

__version__ = "0.1.0"

__all__ = [
	'AgentModule',
	'AgentNetwork',
	'DataPreprocessingAgent',
	'ModelTrainingAgent',
	'HyperparameterAgent',
	'AdaptiveAgent',
	'RewardSystem',
	'AgentZekaLayer',
	'AgentZekaModule',
	'convert_to_agentzeka',
]