# AgentZeka: Agent-Based Neural Network System

AgentZeka, PyTorch kütüphanesine entegre edilmiş, geleneksel katman tabanlı yapay sinir ağlarını agent (ajan) tabanlı bir mimariye dönüştüren yenilikçi bir sistemdir.

## Dokümantasyon

- [API Referansı](API.md)
- [Detaylı Örnekler](EXAMPLES.md)
- [Implementasyon Kılavuzu](IMPLEMENTATION_GUIDE.md)

## Özellikler

- Agent tabanlı yapay sinir ağı mimarisi
- Dinamik görev atama ve yönetimi
- Uzman agent'lar ile özelleştirilmiş işlemler
- PyTorch ile tam entegrasyon
- Ödül-bazlı öğrenme sistemi

## Kurulum

```bash
pip install torch
```

## Hızlı Başlangıç

```python
import torch
import torch.nn as nn
from torch.nn.modules.agentzeka import convert_to_agentzeka

# Geleneksel PyTorch modeli
model = nn.Sequential(
	nn.Linear(10, 20),
	nn.ReLU(),
	nn.Linear(20, 5)
)

# AgentZeka modeline dönüştürme
agent_model = convert_to_agentzeka(model)

# Kullanım
x = torch.randn(32, 10)
output = agent_model(x)
```

## Temel Bileşenler

1. **AgentModule**
   - Temel agent sınıfı
   - Görev yönetimi
   - İletişim altyapısı

2. **Expert Agents**
   - DataPreprocessingAgent
   - ModelTrainingAgent
   - HyperparameterAgent

3. **Agent Network**
   - Agent'lar arası iletişim
   - Görev dağıtımı
   - Ağ yönetimi

## Örnek Kullanımlar

### 1. Veri Ön İşleme

```python
from torch.nn.modules.agentzeka import DataPreprocessingAgent

preprocessor = DataPreprocessingAgent(input_size=10)
task = {
	"type": "preprocessing",
	"normalize": True,
	"standardize": True
}
preprocessor.assign_task(task)
processed_data = preprocessor(data)
```

### 2. Model Eğitimi

```python
from torch.nn.modules.agentzeka import ModelTrainingAgent

trainer = ModelTrainingAgent(model_size=10)
task = {
	"type": "training",
	"adapt_learning": True,
	"batch_size": 32
}
trainer.assign_task(task)
trained_output = trainer(input_data)
```

## Lisans

MIT License