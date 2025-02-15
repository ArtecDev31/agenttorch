import torch
from torch.nn import Linear, Sequential, ReLU
from .agent_integration import convert_to_agentzeka
from .task_management import TaskManager
from .expert_agents import DataPreprocessingAgent, ModelTrainingAgent

def demo_agentzeka():
	"""Demonstrate AgentZeka capabilities."""
	# Create a traditional PyTorch model
	traditional_model = Sequential(
		Linear(10, 20),
		ReLU(),
		Linear(20, 5)
	)
	
	# Convert to AgentZeka model
	agent_model = convert_to_agentzeka(traditional_model)
	
	# Create sample data
	x = torch.randn(32, 10)  # Batch of 32 samples, 10 features each
	
	# Process data through the agent-enhanced model
	output = agent_model(x)
	
	# Demonstrate agent capabilities
	task_manager = TaskManager()
	network = task_manager.create_expert_network(input_size=10)
	
	# Submit various tasks
	preprocessing_task = {
		"type": "data_preprocessing",
		"normalize": True,
		"standardize": True,
		"required_capabilities": {"data_preprocessing"}
	}
	
	training_task = {
		"type": "model_training",
		"adapt_learning": True,
		"loss": 0.5,
		"required_capabilities": {"model_training"}
	}
	
	# Process tasks
	task_manager.submit_task(preprocessing_task, priority=1)
	task_manager.submit_task(training_task, priority=2)
	task_manager.process_tasks()
	
	return {
		"model_output_shape": output.shape,
		"active_agents": len(network.agents),
		"processed_tasks": len(task_manager.task_history)
	}

if __name__ == "__main__":
	results = demo_agentzeka()
	print("Demo Results:", results)