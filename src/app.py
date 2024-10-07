import streamlit as st
import torch
from model.model import NeuralNetwork
from model.sparse_model import SparseNeuralNetwork
from training.training import train_model
from utils.types import NetworkConfig
from visualizer.visualizer import (
    plot_metrics_history, 
    plot_gradient_flow_distribution, 
    plot_network_architecture, 
    plot_gradient_flow,
    display_gradient_flow_stats
)
from data.dataloader import DataLoaderFactory
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import train_test_split
from utils.config import CONFIG, ACTIVATION_FN_MAP

def main():
    st.sidebar.title("Neural Network Settings")

    # Adjust slider precision
    num_layers = st.sidebar.slider("Number of Layers", **CONFIG["num_layers"])
    hidden_size = st.sidebar.slider("Hidden Layer Size", **CONFIG["hidden_size"])
    activation_fn = st.sidebar.selectbox("Activation Function", CONFIG["activation_functions"])
    learning_rate = st.sidebar.number_input("Learning Rate", **CONFIG["learning_rate"])
    num_epochs = st.sidebar.slider("Epochs", **CONFIG["num_epochs"])
    batch_size = st.sidebar.slider("Batch Size", **CONFIG["batch_size"])
    dataset_choice = st.sidebar.selectbox("Dataset", CONFIG["datasets"])
    gradient_clip = st.sidebar.number_input("Gradient Clipping", **CONFIG["gradient_clip"])
    gradient_norm = st.sidebar.number_input("Gradient Normalization", **CONFIG["gradient_norm"])

    activation_fn_enum = ACTIVATION_FN_MAP[activation_fn]

    # Load dataset and split into train and validation sets
    data_loader_factory = DataLoaderFactory(dataset_name=dataset_choice, batch_size=batch_size)
    full_dataloader = data_loader_factory.get_dataloader()
    input_size = data_loader_factory.input_size
    is_regression = data_loader_factory.is_regression

    # Split into train and validation datasets
    train_size = int(0.8 * len(full_dataloader.dataset))
    val_size = len(full_dataloader.dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataloader.dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    config = NetworkConfig(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        activation_fn=activation_fn_enum,
        learning_rate=learning_rate,
        num_epochs=num_epochs
    )

    config.is_regression = is_regression
    config.output_size = len(set(full_dataloader.dataset.tensors[1].numpy())) if not is_regression else 1

    st.title("Interactive Gradient Flow Visualization")

    # Display the neural network architecture
    st.subheader("Dense Neural Network Architecture")
    plot_network_architecture(config)

    st.subheader("Sparse Neural Network Architecture")
    plot_network_architecture(config, sparse=True)

    # Create models and optimizers
    dense_model = NeuralNetwork(config)
    sparse_model = SparseNeuralNetwork(config)
    optimizer_dense = optim.Adam(dense_model.parameters(), lr=learning_rate)
    optimizer_sparse = optim.Adam(sparse_model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss() if is_regression else nn.CrossEntropyLoss()

    if st.button("Start Training"):
        st.subheader("Dense Neural Network Training")
        train_loss_history_dense, val_loss_history_dense, metrics_history_dense, gradient_flow_dense = train_model(
            dense_model, optimizer_dense, loss_fn, config, train_loader, val_loader, is_regression,
            lambda epoch, train_loss, val_loss, metrics: st.sidebar.text(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Metrics: {metrics}"
            ),
            gradient_clip=gradient_clip,
            gradient_norm=gradient_norm
        )

        st.subheader("Sparse Neural Network Training")
        train_loss_history_sparse, val_loss_history_sparse, metrics_history_sparse, gradient_flow_sparse = train_model(
            sparse_model, optimizer_sparse, loss_fn, config, train_loader, val_loader, is_regression,
            lambda epoch, train_loss, val_loss, metrics: st.sidebar.text(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Metrics: {metrics}"
            ),
            gradient_clip=gradient_clip,
            gradient_norm=gradient_norm
        )

        # Plot the train/validation loss and gradient flow distribution for dense model
        st.subheader("Dense Neural Network Metrics")
        plot_metrics_history(metrics_history_dense)
        plot_gradient_flow_distribution(gradient_flow_dense)
        plot_gradient_flow(gradient_flow_dense)
        display_gradient_flow_stats(gradient_flow_dense)

        # Plot the train/validation loss and gradient flow distribution for sparse model
        st.subheader("Sparse Neural Network Metrics")
        plot_metrics_history(metrics_history_sparse)
        plot_gradient_flow_distribution(gradient_flow_sparse)
        plot_gradient_flow(gradient_flow_sparse)
        display_gradient_flow_stats(gradient_flow_sparse)

if __name__ == "__main__":
    main()