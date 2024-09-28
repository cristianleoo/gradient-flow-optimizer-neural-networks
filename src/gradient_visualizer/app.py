import streamlit as st
import torch
from .model import NeuralNetwork
from .training import train_model
from .types import NetworkConfig, ActivationFunction
from .visualizer import plot_loss_history, plot_gradient_flow, plot_network_architecture
import torch.optim as optim
import torch.nn as nn

def main():
    st.sidebar.title("Neural Network Settings")
    num_layers = st.sidebar.slider("Number of Layers", 1, 5, 3)
    hidden_size = st.sidebar.slider("Hidden Layer Size", 10, 100, 50)
    activation_fn = st.sidebar.selectbox("Activation Function", ["ReLU", "Sigmoid", "Tanh"])
    learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.1, 0.001)
    num_epochs = st.sidebar.slider("Epochs", 1, 100, 20)

    activation_fn_enum = ActivationFunction[activation_fn.upper()]

    config = NetworkConfig(
        input_size=10,
        hidden_size=hidden_size,
        num_layers=num_layers,
        activation_fn=activation_fn_enum,
        learning_rate=learning_rate,
        num_epochs=num_epochs
    )

    st.title("Interactive Gradient Flow Visualization")

    # Visualize network architecture
    plot_network_architecture(config)

    # Generate synthetic data
    X = torch.randn(100, 10)
    y = (X.sum(dim=1) > 0).long()

    # Create model and optimizer
    model = NeuralNetwork(config)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    if st.button("Start Training"):
        loss_history, gradient_flow = train_model(model, optimizer, loss_fn, config, X, y, lambda epoch, loss: st.sidebar.text(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}"))

        plot_loss_history(loss_history)
        plot_gradient_flow(gradient_flow)

if __name__ == "__main__":
    main()
