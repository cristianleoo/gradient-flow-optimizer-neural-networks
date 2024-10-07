import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataLoaderFactory:
    def __init__(self, dataset_name: str, batch_size: int = 32):
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.input_size = None
        self.is_regression = False

    def get_dataloader(self):
        # Load dataset
        if self.dataset_name == "Diabetes":
            data = datasets.load_diabetes()
            self.is_regression = True
        elif self.dataset_name == "Iris":
            data = datasets.load_iris()
        elif self.dataset_name == "Wine":
            data = datasets.load_wine()
        elif self.dataset_name == "Breast Cancer":
            data = datasets.load_breast_cancer()
        else:
            raise ValueError("Unsupported dataset.")
        
        X, y = data.data, data.target
        self.input_size = X.shape[1]  # Set input size based on dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32 if self.is_regression else torch.long)

        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
