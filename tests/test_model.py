import pytest
import torch
from model.model import NeuralNetwork

def test_neural_network():
    model = NeuralNetwork(input_size=10, hidden_size=50, num_layers=3, activation_fn='ReLU')
    X = torch.randn(5, 10)
    output = model(X)
    assert output.shape == (5, 2), "Output shape mismatch"
