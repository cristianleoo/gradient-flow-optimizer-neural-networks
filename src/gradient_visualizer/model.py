import torch.nn as nn
from abc import ABC, abstractmethod
from .types import ActivationFunction, NetworkConfig

class BaseModel(ABC):
    @abstractmethod
    def forward(self, x):
        pass

class NeuralNetwork(BaseModel, nn.Module):
    def __init__(self, config: NetworkConfig):
        super(NeuralNetwork, self).__init__()
        layers = []
        for i in range(config.num_layers):
            layers.append(nn.Linear(config.input_size if i == 0 else config.hidden_size, config.hidden_size))
            layers.append(self.get_activation_fn(config.activation_fn))
        layers.append(nn.Linear(config.hidden_size, 2))  # Output layer
        self.model = nn.Sequential(*layers)

    def get_activation_fn(self, activation_fn: ActivationFunction):
        if activation_fn == ActivationFunction.RELU:
            return nn.ReLU()
        elif activation_fn == ActivationFunction.SIGMOID:
            return nn.Sigmoid()
        else:
            return nn.Tanh()

    def forward(self, x):
        return self.model(x)
