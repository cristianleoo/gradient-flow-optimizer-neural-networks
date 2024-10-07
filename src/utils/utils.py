from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

# Calculate metrics for both train and validation sets
def calculate_metrics(y_true_train, y_pred_train, y_true_val, y_pred_val, is_regression):
    if is_regression:
        train_metrics = {
            'MSE': mean_squared_error(y_true_train, y_pred_train),
            'MAE': mean_absolute_error(y_true_train, y_pred_train),
            'R2': r2_score(y_true_train, y_pred_train)
        }
        val_metrics = {
            'MSE': mean_squared_error(y_true_val, y_pred_val),
            'MAE': mean_absolute_error(y_true_val, y_pred_val),
            'R2': r2_score(y_true_val, y_pred_val)
        }
    else:
        train_metrics = {
            'Accuracy': accuracy_score(y_true_train, y_pred_train),
            'Precision': precision_score(y_true_train, y_pred_train, average='weighted', zero_division=0),
            'Recall': recall_score(y_true_train, y_pred_train, average='weighted', zero_division=0),
            'F1 Score': f1_score(y_true_train, y_pred_train, average='weighted', zero_division=0)
        }
        val_metrics = {
            'Accuracy': accuracy_score(y_true_val, y_pred_val),
            'Precision': precision_score(y_true_val, y_pred_val, average='weighted', zero_division=0),
            'Recall': recall_score(y_true_val, y_pred_val, average='weighted', zero_division=0),
            'F1 Score': f1_score(y_true_val, y_pred_val, average='weighted', zero_division=0)
        }
    
    return train_metrics, val_metrics