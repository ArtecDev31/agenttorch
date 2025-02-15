import unittest
import torch
import os
import sys

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from torch.nn.modules.agentzeka_init import (
	AgentModule,
	DataPreprocessingAgent,
	AdaptiveAgent,
	AgentZekaLayer
)

class TestStandaloneAgentZeka(unittest.TestCase):
	"""Standalone tests for AgentZeka components."""
	
	def setUp(self):
		"""Set up test environment."""
		self.input_size = 10
		self.output_size = 5
		self.batch_size = 32
	
	def test_agent_creation(self):
		"""Test basic agent creation."""
		agent = AgentModule()
		self.assertIsInstance(agent, AgentModule)
		self.assertEqual(len(agent.agent_state), 0)
		
	def test_preprocessing_agent(self):
		"""Test preprocessing agent functionality."""
		agent = DataPreprocessingAgent(self.input_size)
		x = torch.randn(5, self.input_size)
		
		# Test normalization
		output = agent._expert_forward(x, {"normalize": True})
		self.assertEqual(output.shape, x.shape)
		
	def test_adaptive_agent(self):
		"""Test adaptive agent learning."""
		agent = AdaptiveAgent(self.input_size)
		x = torch.randn(5, self.input_size)
		
		# Test adaptation
		metrics = {"task_completion": 0.8, "efficiency": 0.7}
		agent.adapt(x, metrics)
		self.assertGreater(len(agent.state_history), 0)
		
	def test_agent_layer(self):
		"""Test AgentZeka layer integration."""
		layer = AgentZekaLayer(self.input_size, self.output_size)
		x = torch.randn(3, self.input_size)
		
		output = layer(x)
		self.assertEqual(output.shape, (3, self.output_size))

if __name__ == '__main__':
	unittest.main()