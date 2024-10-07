from dataclasses import dataclass
from enum import Enum

from enum import Enum

class ActivationFunction(Enum):
    RELU = "ReLU"
    SIGMOID = "Sigmoid"
    TANH = "Tanh"
    SWISH = "Swish"
    PRELU = "PReLU"

@dataclass
class NetworkConfig:
    input_size: int
    hidden_size: int
    num_layers: int
    activation_fn: ActivationFunction
    learning_rate: float
    num_epochs: int
