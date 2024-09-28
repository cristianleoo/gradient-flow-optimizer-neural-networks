import plotly.graph_objects as go
import streamlit as st

def plot_loss_history(loss_history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=loss_history, mode='lines', name='Loss'))
    fig.update_layout(title="Loss Over Time", xaxis_title="Epoch", yaxis_title="Loss")
    st.plotly_chart(fig)

def plot_gradient_flow(gradient_flow):
    fig = go.Figure()
    for i, layer_gradients in enumerate(zip(*gradient_flow)):
        fig.add_trace(go.Scatter(y=layer_gradients, mode='lines', name=f'Layer {i + 1}'))
    fig.update_layout(title="Gradient Flow Across Layers", xaxis_title="Epoch", yaxis_title="Gradient Norm")
    st.plotly_chart(fig)

def plot_network_architecture(config):
    nodes_per_layer = [config.input_size] + [config.hidden_size] * config.num_layers + [2]
    fig = go.Figure()

    for layer_idx, nodes in enumerate(nodes_per_layer):
        for i in range(nodes):
            fig.add_trace(go.Scatter(
                x=[layer_idx], y=[i], mode='markers', marker=dict(size=20),
                name=f'Layer {layer_idx+1}', showlegend=(i==0)
            ))
    fig.update_layout(title="Neural Network Architecture", showlegend=False)
    st.plotly_chart(fig)
