import unittest
import torch
import sys
import os

# Add the path to our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'torch', 'nn', 'modules'))

from agentzeka import AgentModule, AgentNetwork
from expert_agents import DataPreprocessingAgent
from agent_learning import AdaptiveAgent
from agent_integration import AgentZekaLayer

class TestAgentZeka(unittest.TestCase):
	def setUp(self):
		self.input_size = 10
		self.output_size = 5
		
	def test_basic_functionality(self):
		print("\nTesting basic agent functionality...")
		agent = AgentModule()
		self.assertIsInstance(agent, AgentModule)
		print("✓ Basic agent creation successful")
		
	def test_preprocessing(self):
		print("\nTesting preprocessing agent...")
		agent = DataPreprocessingAgent(self.input_size)
		x = torch.randn(5, self.input_size)
		output = agent._expert_forward(x, {"normalize": True})
		self.assertEqual(output.shape, x.shape)
		print("✓ Preprocessing test successful")
		
	def test_adaptive_learning(self):
		print("\nTesting adaptive learning...")
		agent = AdaptiveAgent(self.input_size)
		x = torch.randn(5, self.input_size)
		metrics = {"task_completion": 0.8}
		agent.adapt(x, metrics)
		print("✓ Adaptive learning test successful")
		
	def test_integration(self):
		print("\nTesting integration layer...")
		layer = AgentZekaLayer(self.input_size, self.output_size)
		x = torch.randn(3, self.input_size)
		output = layer(x)
		self.assertEqual(output.shape, (3, self.output_size))
		print("✓ Integration test successful")

if __name__ == '__main__':
	print("Starting AgentZeka tests...")
	unittest.main(verbosity=2)