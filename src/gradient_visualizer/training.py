import torch
from .model import BaseModel
from .types import NetworkConfig

def train_model(model: BaseModel, optimizer, loss_fn, config: NetworkConfig, X, y, progress_callback):
    loss_history = []
    gradient_flow = []

    for epoch in range(config.num_epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()

        grads = [param.grad.norm().item() for param in model.parameters() if param.grad is not None]
        gradient_flow.append(grads)

        optimizer.step()
        loss_history.append(loss.item())

        # Call the progress callback to update Streamlit
        progress_callback(epoch, loss.item())

    return loss_history, gradient_flow
