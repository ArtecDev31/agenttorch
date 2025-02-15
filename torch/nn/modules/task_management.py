import torch
from typing import Dict, List, Any, Optional, Set
from .agentzeka import AgentModule, AgentNetwork
from .expert_agents import DataPreprocessingAgent, ModelTrainingAgent, HyperparameterAgent
import threading
import queue

class TaskManager:
	"""Manages task distribution and coordination between agents."""
	
	def __init__(self):
		self.task_queue = queue.PriorityQueue()
		self.agent_network = AgentNetwork()
		self.task_history = []
		self.active_tasks = set()
		self.agent_capabilities = {}
		
	def register_agent_capability(self, agent_name: str, capabilities: Set[str]) -> None:
		"""Register an agent's capabilities for task matching."""
		self.agent_capabilities[agent_name] = capabilities
		
	def submit_task(self, task: Dict[str, Any], priority: int = 0) -> None:
		"""Submit a task to be processed by agents."""
		self.task_queue.put((priority, task))
		
	def assign_task(self, task: Dict[str, Any]) -> Optional[str]:
		"""Assign task to the most suitable agent."""
		task_type = task.get("type", "general")
		best_agent = None
		max_capability_match = 0
		
		for agent_name, capabilities in self.agent_capabilities.items():
			if task_type in capabilities:
				capability_match = len(capabilities.intersection(task.get("required_capabilities", set())))
				if capability_match > max_capability_match:
					max_capability_match = capability_match
					best_agent = agent_name
					
		if best_agent:
			self.agent_network.agents[best_agent].assign_task(task)
			self.active_tasks.add(id(task))
		return best_agent
		
	def process_tasks(self) -> None:
		"""Process all tasks in the queue."""
		while not self.task_queue.empty():
			priority, task = self.task_queue.get()
			agent_name = self.assign_task(task)
			if agent_name:
				self.task_history.append({
					"task": task,
					"agent": agent_name,
					"priority": priority,
					"status": "assigned"
				})
				
	def create_expert_network(self, input_size: int) -> AgentNetwork:
		"""Create a network of expert agents."""
		network = AgentNetwork()
		
		# Create expert agents
		preprocessor = DataPreprocessingAgent(input_size)
		trainer = ModelTrainingAgent(input_size)
		hypertuner = HyperparameterAgent(input_size)
		
		# Add agents to network
		network.add_agent("preprocessor", preprocessor)
		network.add_agent("trainer", trainer)
		network.add_agent("hypertuner", hypertuner)
		
		# Connect agents
		network.connect_agents("preprocessor", "trainer")
		network.connect_agents("trainer", "hypertuner")
		
		# Register capabilities
		self.register_agent_capability("preprocessor", {"data_preprocessing", "normalization"})
		self.register_agent_capability("trainer", {"model_training", "optimization"})
		self.register_agent_capability("hypertuner", {"hyperparameter_optimization", "tuning"})
		
		self.agent_network = network
		return network

class TaskScheduler:
	"""Schedules and prioritizes tasks for the agent network."""
	
	def __init__(self, task_manager: TaskManager):
		self.task_manager = task_manager
		self.scheduling_thread = None
		self.running = False
		
	def start(self):
		"""Start the task scheduling thread."""
		self.running = True
		self.scheduling_thread = threading.Thread(target=self._scheduling_loop)
		self.scheduling_thread.start()
		
	def stop(self):
		"""Stop the task scheduling thread."""
		self.running = False
		if self.scheduling_thread:
			self.scheduling_thread.join()
			
	def _scheduling_loop(self):
		"""Main scheduling loop."""
		while self.running:
			self.task_manager.process_tasks()
			# Add small delay to prevent CPU overuse
			threading.Event().wait(0.1)