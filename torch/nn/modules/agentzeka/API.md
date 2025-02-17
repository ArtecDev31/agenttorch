# AgentZeka API Dokümantasyonu

## Core API

### AgentModule

```python
class AgentModule(nn.Module):
	"""
	Temel agent sınıfı.
	
	Özellikler:
		agent_state (Dict): Agent durumu
		task_queue (Queue): Görev kuyruğu
		message_queue (Queue): İletişim kuyruğu
	"""
	
	def assign_task(self, task: Dict[str, Any]) -> None:
		"""Görevi agent'a atar"""
	
	def send_message(self, target: str, message: Dict) -> None:
		"""Diğer agent'lara mesaj gönderir"""
```

### Expert Agents

#### DataPreprocessingAgent

```python
class DataPreprocessingAgent(AgentModule):
	"""
	Veri ön işleme uzmanı.
	
	Özellikler:
		input_size (int): Giriş boyutu
		normalization_layer: Normalizasyon katmanı
	"""
```

#### ModelTrainingAgent

```python
class ModelTrainingAgent(AgentModule):
	"""
	Model eğitim uzmanı.
	
	Özellikler:
		model_size (int): Model boyutu
		optimization_history: Optimizasyon geçmişi
	"""
```

### Integration API

#### convert_to_agentzeka

```python
def convert_to_agentzeka(model: nn.Module) -> AgentModule:
	"""
	PyTorch modelini AgentZeka modeline dönüştürür.
	
	Args:
		model: Dönüştürülecek PyTorch modeli
		
	Returns:
		AgentZeka modeli
	"""
```

## Görev Formatları

### Preprocessing Task
```python
{
	"type": "preprocessing",
	"normalize": bool,
	"standardize": bool
}
```

### Training Task
```python
{
	"type": "training",
	"adapt_learning": bool,
	"batch_size": int
}
```

### Hyperparameter Task
```python
{
	"type": "hyperparameter_optimization",
	"tune_params": bool,
	"param_space": Dict
}
```