import unittest
import torch
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentzeka.core import AgentModule, AgentNetwork
from agentzeka.experts import DataPreprocessingAgent, ModelTrainingAgent
from agentzeka.learning import AdaptiveAgent
from agentzeka.integration import AgentZekaLayer, convert_to_agentzeka

class TestAgentZeka(unittest.TestCase):
	def setUp(self):
		self.input_size = 10
		self.output_size = 5
		self.batch_size = 32
		
	def test_agent_module(self):
		print("\nTesting AgentModule...")
		agent = AgentModule()
		self.assertIsInstance(agent, AgentModule)
		self.assertEqual(len(agent.agent_state), 0)
		print("✓ AgentModule test passed")
		
	def test_preprocessing_agent(self):
		print("\nTesting PreprocessingAgent...")
		agent = DataPreprocessingAgent(self.input_size)
		x = torch.randn(self.batch_size, self.input_size)
		output = agent._expert_forward(x, {"normalize": True})
		self.assertEqual(output.shape, x.shape)
		print("✓ PreprocessingAgent test passed")
		
	def test_adaptive_agent(self):
		print("\nTesting AdaptiveAgent...")
		agent = AdaptiveAgent(self.input_size)
		x = torch.randn(self.batch_size, self.input_size)
		metrics = {"task_completion": 0.8, "efficiency": 0.7}
		agent.adapt(x, metrics)
		self.assertGreater(len(agent.state_history), 0)
		print("✓ AdaptiveAgent test passed")
		
	def test_model_conversion(self):
		print("\nTesting model conversion...")
		model = torch.nn.Sequential(
			torch.nn.Linear(self.input_size, 20),
			torch.nn.Linear(20, self.output_size)
		)
		agent_model = convert_to_agentzeka(model)
		x = torch.randn(self.batch_size, self.input_size)
		output = agent_model(x)
		self.assertEqual(output.shape, (self.batch_size, self.output_size))
		print("✓ Model conversion test passed")

if __name__ == '__main__':
	print("Starting AgentZeka tests...")
	unittest.main(verbosity=2)