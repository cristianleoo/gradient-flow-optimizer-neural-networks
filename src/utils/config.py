from utils.types import ActivationFunction

# Configuration for the application
CONFIG = {
    "num_layers": {"min_value": 1, "max_value": 10, "value": 3, "step": 1},
    "hidden_size": {"min_value": 1, "max_value": 500, "value": 10, "step": 1},
    "activation_functions": ["ReLU", "Sigmoid", "Tanh", "Swish", "PReLU"],
    "learning_rate": {"min_value": 0.0001, "max_value": 1.0, "value": 0.001, "step": 0.0001, "format": "%.5f"},
    "num_epochs": {"min_value": 1, "max_value": 200, "value": 20, "step": 1},
    "batch_size": {"min_value": 1, "max_value": 256, "value": 32, "step": 1},
    "datasets": ["Diabetes", "Iris", "Wine", "Breast Cancer"],
    "gradient_clip": {"min_value": 0.0, "max_value": 5.0, "value": 0.0, "step": 0.1, "format": "%.2f"},
    "gradient_norm": {"min_value": 0.0, "max_value": 5.0, "value": 0.0, "step": 0.1, "format": "%.2f"}
}

# Mapping activation function names to their enum values
ACTIVATION_FN_MAP = {name: ActivationFunction[name.upper()] for name in CONFIG["activation_functions"]}