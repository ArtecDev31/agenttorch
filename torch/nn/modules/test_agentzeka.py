import torch
import unittest
from torch.nn import Linear, Sequential
from .agentzeka import AgentModule, AgentNetwork
from .expert_agents import DataPreprocessingAgent, ModelTrainingAgent, HyperparameterAgent
from .agent_integration import AgentZekaLayer, AgentZekaModule, convert_to_agentzeka

class TestAgentZeka(unittest.TestCase):
	def setUp(self):
		self.input_size = 10
		self.output_size = 5
		self.batch_size = 32
		
	def test_agent_module(self):
		"""Test basic agent module functionality."""
		agent = AgentModule()
		self.assertIsNotNone(agent.agent_state)
		self.assertIsNotNone(agent.task_queue)
		
	def test_expert_agents(self):
		"""Test expert agents initialization and processing."""
		preprocessor = DataPreprocessingAgent(self.input_size)
		trainer = ModelTrainingAgent(self.input_size)
		hypertuner = HyperparameterAgent(self.input_size)
		
		x = torch.randn(self.batch_size, self.input_size)
		
		# Test preprocessing
		task = {"normalize": True, "standardize": True}
		output = preprocessor._expert_forward(x, task)
		self.assertEqual(output.shape, x.shape)
		
	def test_agent_network(self):
		"""Test agent network creation and communication."""
		network = AgentNetwork()
		
		# Add agents
		network.add_agent("preprocessor", DataPreprocessingAgent(self.input_size))
		network.add_agent("trainer", ModelTrainingAgent(self.input_size))
		
		# Test connections
		network.connect_agents("preprocessor", "trainer")
		self.assertIn("preprocessor", network.connections)
		
	def test_integration_layer(self):
		"""Test integration with PyTorch layers."""
		layer = AgentZekaLayer(self.input_size, self.output_size)
		x = torch.randn(self.batch_size, self.input_size)
		output = layer(x)
		self.assertEqual(output.shape, (self.batch_size, self.output_size))
		
	def test_model_conversion(self):
		"""Test conversion of traditional model to AgentZeka."""
		# Create traditional model
		model = Sequential(
			Linear(self.input_size, 20),
			Linear(20, self.output_size)
		)
		
		# Convert to AgentZeka
		agent_model = convert_to_agentzeka(model)
		
		# Test forward pass
		x = torch.randn(self.batch_size, self.input_size)
		output = agent_model(x)
		self.assertEqual(output.shape, (self.batch_size, self.output_size))

if __name__ == '__main__':
	unittest.main()