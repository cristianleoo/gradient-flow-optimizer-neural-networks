import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, activation_fn):
        super(NeuralNetwork, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_size if i == 0 else hidden_size, hidden_size))
            if activation_fn == "ReLU":
                layers.append(nn.ReLU())
            elif activation_fn == "Sigmoid":
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, 2))  # Output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
