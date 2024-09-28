import torch

def train_model(model, optimizer, loss_fn, num_epochs, X, y, progress_bar, status_text):
    loss_history = []
    gradient_flow = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()

        # Capture gradients
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.norm().item())
        gradient_flow.append(grads)

        optimizer.step()

        # Record loss
        loss_history.append(loss.item())

        # Update progress bar
        progress_bar.progress((epoch + 1) / num_epochs)
        status_text.text(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    return loss_history, gradient_flow
