import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import torch

def plot_metrics_history(metrics_history):
    # Extract metric names from the first epoch's metrics
    metrics = list(metrics_history['train'][0].keys())
    tabs = st.tabs(metrics)

    for i, metric in enumerate(metrics):
        with tabs[i]:
            train_metric_values = [epoch_metrics[metric] for epoch_metrics in metrics_history['train']]
            val_metric_values = [epoch_metrics[metric] for epoch_metrics in metrics_history['val']]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=train_metric_values, mode='lines', name='Train ' + metric))
            fig.add_trace(go.Scatter(y=val_metric_values, mode='lines', name='Validation ' + metric))
            fig.update_layout(title=f"{metric} Over Time", xaxis_title="Epoch", yaxis_title=metric)
            st.plotly_chart(fig)

def plot_gradient_flow_distribution(gradient_flow):
    fig = go.Figure()

    # Create distribution for each layer
    for layer_idx, gradients in enumerate(zip(*gradient_flow)):
        fig.add_trace(go.Box(y=gradients, name=f'Layer {layer_idx + 1}'))

    fig.update_layout(title="Gradient Flow Distribution Across Layers", xaxis_title="Layer", yaxis_title="Gradient Norm")
    st.plotly_chart(fig)

def plot_gradient_flow(gradient_flow):
    fig = go.Figure()
    for i, layer_gradients in enumerate(zip(*gradient_flow)):
        fig.add_trace(go.Scatter(y=layer_gradients, mode='lines', name=f'Layer {i + 1}'))
    fig.update_layout(title="Gradient Flow Across Layers", xaxis_title="Epoch", yaxis_title="Gradient Norm")
    st.plotly_chart(fig)


def plot_network_architecture(config, sparse=False):
    nodes_per_layer = [config.input_size] + [config.hidden_size] * config.num_layers + [1 if config.is_regression else config.output_size]
    max_nodes = max(nodes_per_layer)
    fig = go.Figure()

    # Plot nodes and edges
    for layer_idx, nodes in enumerate(nodes_per_layer):
        for i in range(nodes):
            y_pos = max_nodes / 2 + (i - nodes / 2)
            fig.add_trace(go.Scatter(
                x=[layer_idx], y=[y_pos], mode='markers+text', text=[f'{i}'],
                marker=dict(size=20), showlegend=False
            ))

            if layer_idx > 0:
                prev_nodes = nodes_per_layer[layer_idx - 1]
                for j in range(prev_nodes):
                    prev_y_pos = max_nodes / 2 + (j - prev_nodes / 2)
                    if not sparse or torch.rand(1).item() > 0.5:  # Randomly drop connections for sparse network
                        fig.add_trace(go.Scatter(
                            x=[layer_idx - 1, layer_idx], y=[prev_y_pos, y_pos], mode='lines', line=dict(color='gray'),
                            showlegend=False
                        ))

    fig.update_layout(title="Neural Network Architecture" + (" (Sparse)" if sparse else ""), xaxis=dict(showgrid=False, zeroline=False),
                      yaxis=dict(showgrid=False, zeroline=False), showlegend=False)
    st.plotly_chart(fig)

def display_gradient_flow_stats(gradient_flow):
    # Calculate statistics for each layer
    stats = []
    vanishing_threshold = 1e-5
    exploding_threshold = 1e2

    for layer_idx, gradients in enumerate(zip(*gradient_flow)):
        gradients_series = pd.Series(gradients)
        initial_gradient = gradients_series.iloc[0]
        ending_gradient = gradients_series.iloc[-1]
        num_vanishing = (gradients_series.abs() < vanishing_threshold).sum()
        num_exploding = (gradients_series.abs() > exploding_threshold).sum()

        stats.append({
            'Layer': layer_idx + 1,
            'Initial': round(initial_gradient, 2),
            'Ending': round(ending_gradient, 2),
            'Mean': round(gradients_series.mean(), 2),
            'Std Dev': round(gradients_series.std(), 2),
            'Min': round(gradients_series.min(), 2),
            '25%': round(gradients_series.quantile(0.25), 2),
            '50%': round(gradients_series.median(), 2),
            '75%': round(gradients_series.quantile(0.75), 2),
            'Max': round(gradients_series.max(), 2),
            'Vanishing': num_vanishing,
            'Exploding': num_exploding
        })

    # Create a DataFrame and display it
    df_stats = pd.DataFrame(stats)
    st.table(df_stats)