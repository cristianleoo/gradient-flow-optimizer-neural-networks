import torch
from utils.utils import calculate_metrics

def train_model(model, optimizer, loss_fn, config, train_loader, val_loader, is_regression, progress_callback, gradient_clip=None, gradient_norm=None):
    train_loss_history = []
    val_loss_history = []
    metrics_history = {
        'train': [],
        'val': []
    }
    gradient_flow = []

    for epoch in range(config.num_epochs):
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0
        y_true_train, y_pred_train = [], []
        y_true_val, y_pred_val = [], []

        # Training phase
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()

            if is_regression:
                y_batch = y_batch.view_as(outputs)
                loss = loss_fn(outputs, y_batch)
                preds = outputs.detach().numpy()

                if y_batch.dim() == 0:
                    y_true_train.append(y_batch.item())
                else:
                    y_true_train.extend(y_batch.detach().numpy().tolist())
                y_pred_train.extend(preds.tolist())
            else:
                loss = loss_fn(outputs, y_batch.long())
                preds = torch.argmax(outputs, dim=1).detach().numpy()

                y_true_train.extend(y_batch.numpy().tolist())
                y_pred_train.extend(preds.tolist())

            loss.backward()

            # Gradient clipping or normalization
            if gradient_clip and gradient_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            if gradient_norm and gradient_norm > 0.0:
                torch.nn.utils.clip_grad_value_(model.parameters(), gradient_norm)

            grads = [param.grad.norm().item() for param in model.parameters() if param.grad is not None]
            gradient_flow.append(grads)

            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_loss_history.append(avg_train_loss)

        # Validation phase
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch).squeeze()

                if is_regression:
                    y_batch = y_batch.view_as(outputs)
                    loss = loss_fn(outputs, y_batch)
                    preds = outputs.detach().numpy()

                    if y_batch.dim() == 0:
                        y_true_val.append(y_batch.item())
                    else:
                        y_true_val.extend(y_batch.detach().numpy().tolist())
                    y_pred_val.extend(preds.tolist())
                else:
                    loss = loss_fn(outputs, y_batch.long())
                    preds = torch.argmax(outputs, dim=1).detach().numpy()

                    y_true_val.extend(y_batch.numpy().tolist())
                    y_pred_val.extend(preds.tolist())

                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        # Calculate metrics (for train set)
        # if is_regression:
        #     mse = mean_squared_error(y_true_train, y_pred_train)
        #     mae = mean_absolute_error(y_true_train, y_pred_train)
        #     r2 = r2_score(y_true_train, y_pred_train)
        #     metrics = {'MSE': mse, 'MAE': mae, 'R2': r2}
        # else:
        #     accuracy = accuracy_score(y_true_train, y_pred_train)
        #     precision = precision_score(y_true_train, y_pred_train, average='weighted', zero_division=0)
        #     recall = recall_score(y_true_train, y_pred_train, average='weighted', zero_division=0)
        #     f1 = f1_score(y_true_train, y_pred_train, average='weighted', zero_division=0)
        #     metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
        train_metrics, val_metrics = calculate_metrics(y_true_train, y_pred_train, y_true_val, y_pred_val, is_regression)
        metrics_history['train'].append(train_metrics)
        metrics_history['val'].append(val_metrics)

        # metric_history.append(metrics)

        # Update progress
        progress_callback(epoch, avg_train_loss, avg_val_loss, metrics_history)

    return train_loss_history, val_loss_history, metrics_history, gradient_flow
