import streamlit as st
from model import NeuralNetwork
from training import train_model
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Sidebar for neural network parameters
    st.sidebar.title("Neural Network Settings")
    num_layers = st.sidebar.slider("Number of Layers", 1, 5, 3)
    hidden_size = st.sidebar.slider("Hidden Layer Size", 10, 100, 50)
    activation_fn = st.sidebar.selectbox("Activation Function", ["ReLU", "Sigmoid", "Tanh"])
    learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.1, 0.001)
    num_epochs = st.sidebar.slider("Epochs", 1, 100, 20)

    # Placeholder for live updates
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    # Main Title
    st.title("Interactive Gradient Flow Visualization")

    # Sample data for training
    X = torch.randn(100, 10)
    y = (X.sum(dim=1) > 0).long()

    # Instantiate model, optimizer, and loss function
    model = NeuralNetwork(input_size=10, hidden_size=hidden_size, num_layers=num_layers, activation_fn=activation_fn)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Start Training Button
    if st.button("Start Training"):
        loss_history, gradient_flow = train_model(model, optimizer, loss_fn, num_epochs, X, y, progress_bar, status_text)

        # Plot loss history
        st.subheader("Loss Over Time")
        plt.figure(figsize=(10, 4))
        plt.plot(loss_history)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        st.pyplot(plt)

        # Gradient Flow Visualization
        st.subheader("Gradient Flow Across Layers")
        gradient_flow = np.array(gradient_flow).T
        plt.figure(figsize=(10, 6))
        for i in range(len(gradient_flow)):
            plt.plot(gradient_flow[i], label=f"Layer {i + 1}")
        plt.xlabel("Epoch")
        plt.ylabel("Gradient Norm")
        plt.legend()
        st.pyplot(plt)

if __name__ == "__main__":
    main()
