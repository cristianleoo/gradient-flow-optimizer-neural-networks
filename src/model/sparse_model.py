import torch.nn as nn
from model.base.base import BaseModel
from utils.types import ActivationFunction, NetworkConfig

class SparseNeuralNetwork(BaseModel, nn.Module):
    def __init__(self, config: NetworkConfig):
        super(SparseNeuralNetwork, self).__init__()
        layers = []
        input_size = config.input_size
        for i in range(config.num_layers):
            layers.append(nn.Linear(input_size if i == 0 else config.hidden_size, config.hidden_size, bias=False))
            layers.append(nn.Dropout(0.5))  # Example of sparsity
            layers.append(self.get_activation_fn(config.activation_fn))
        layers.append(nn.Linear(config.hidden_size, 1 if config.is_regression else config.output_size))
        self.model = nn.Sequential(*layers)

    def get_activation_fn(self, activation_fn: ActivationFunction):
        if activation_fn == ActivationFunction.RELU:
            return nn.ReLU()
        elif activation_fn == ActivationFunction.SIGMOID:
            return nn.Sigmoid()
        elif activation_fn == ActivationFunction.SWISH:
            return nn.SiLU()  # Swish is implemented as SiLU in PyTorch
        elif activation_fn == ActivationFunction.PRELU:
            return nn.PReLU()
        else:
            return nn.Tanh()

    def forward(self, x):
        return self.model(x)