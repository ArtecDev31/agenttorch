# AgentZeka Örnek Kullanımlar

## 1. Temel Kullanım

```python
import torch
from torch.nn.modules.agentzeka import convert_to_agentzeka
import torch.nn as nn

# Model oluştur
model = nn.Sequential(
	nn.Linear(10, 20),
	nn.ReLU(),
	nn.Linear(20, 5)
)

# AgentZeka'ya dönüştür
agent_model = convert_to_agentzeka(model)

# Kullan
x = torch.randn(32, 10)
output = agent_model(x)
```

## 2. Özel Agent Oluşturma

```python
from torch.nn.modules.agentzeka import AgentModule

class CustomAgent(AgentModule):
	def __init__(self, input_size):
		super().__init__()
		self.input_size = input_size
		self.specialize("custom_domain")
		
	def _expert_forward(self, x, task):
		# Özel işleme mantığı
		return x
```

## 3. Agent Ağı Oluşturma

```python
from torch.nn.modules.agentzeka import AgentNetwork
from torch.nn.modules.agentzeka import DataPreprocessingAgent

# Ağ oluştur
network = AgentNetwork()

# Agent'ları ekle
preprocessor = DataPreprocessingAgent(input_size=10)
network.add_agent("preprocessor", preprocessor)

# Görev tanımla ve işle
task = {
	"type": "preprocessing",
	"normalize": True
}
preprocessor.assign_task(task)

# Veriyi işle
output = network(input_data)
```

## 4. Eğitim Örneği

```python
from torch.nn.modules.agentzeka import ModelTrainingAgent
import torch.optim as optim

# Eğitim agent'ı
trainer = ModelTrainingAgent(model_size=10)

# Optimizer ve loss
optimizer = optim.Adam(trainer.parameters())
criterion = nn.MSELoss()

# Eğitim döngüsü
for epoch in range(10):
	optimizer.zero_grad()
	
	# Görev tanımla
	task = {
		"type": "training",
		"adapt_learning": True,
		"batch_size": 32
	}
	trainer.assign_task(task)
	
	# Forward ve backward
	output = trainer(input_data)
	loss = criterion(output, target)
	loss.backward()
	optimizer.step()
```

## 5. Performans İzleme

```python
# Agent durumunu kontrol et
print(f"Görev kuyruğu boyutu: {len(agent.task_queue.queue)}")
print(f"Agent durumu: {agent.agent_state}")
print(f"Ödül geçmişi: {agent.reward_history}")
```