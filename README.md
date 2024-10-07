# Gradient Flow Visualizer

This package provides an interactive Streamlit app to visualize the gradient flow of a simple neural network during training. You can adjust the network's parameters and see real-time updates of the loss and gradient magnitudes.

## Features

- **Neural Network Settings**: Configure the number of layers, hidden size, activation function, and learning rate.
- **Live Gradient Flow**: Visualize the gradient norms across different layers during training.
- **Real-time Feedback**: See the loss curve and how gradients change over time as you adjust parameters.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/gradient-visualizer.git
cd gradient-visualizer
pip install -r requirements.txt

````


# Gradient Flow Visualizer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-red)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-orange)](https://streamlit.io/)

An interactive Streamlit application for visualizing gradient flow in neural networks during training. This tool allows researchers and practitioners to gain insights into the behavior of gradients across different network architectures and hyperparameters.

## 🚀 Features

- **Customizable Neural Network Settings**: Configure network depth, width, activation functions, and learning rates.
- **Real-time Gradient Flow Visualization**: Observe gradient norms across different layers during training.
- **Performance Metrics**: Track loss curves and various performance metrics as you adjust parameters.
- **Support for Dense and Sparse Networks**: Compare gradient flow between dense and sparse architectures.
- **Multiple Datasets**: Choose from a variety of built-in datasets for experimentation.
- **Advanced Training Options**: Implement gradient clipping and normalization techniques.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/gradient-flow-visualizer.git
   cd gradient-flow-visualizer
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🖥️ Usage

To run the Gradient Flow Visualizer:

```bash
streamlit run src/app.py
```

Navigate to the provided local URL in your web browser to interact with the application.

## 🧠 How It Works

The Gradient Flow Visualizer leverages Streamlit for its interactive interface and PyTorch for neural network training. The main components include:

1. **Neural Network Models**: Implemented in `src/model/model.py` and `src/model/sparse_model.py`.
2. **Training Loop**: Defined in `src/training/training.py`.
3. **Data Loading**: Handled by `src/data/dataloader.py`.
4. **Visualization**: Powered by `src/visualizer/visualizer.py`.

The application allows users to configure network parameters, select datasets, and visualize gradient flow in real-time during training.

## 📊 Interpreting Results

The Gradient Flow Visualizer provides several key visualizations:

1. **Neural Network Architecture**: Visual representation of the network structure.
2. **Metrics History**: Plots of various performance metrics over training epochs.
3. **Gradient Flow Distribution**: Box plots showing the distribution of gradient norms across layers.
4. **Gradient Flow Over Time**: Line plots depicting how gradient norms change during training.
5. **Gradient Flow Statistics**: Detailed statistics on gradient behavior for each layer.

These visualizations help identify issues like vanishing or exploding gradients and assess the overall health of gradient flow in the network.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For any questions or feedback, please open an issue in the GitHub repository or contact the maintainer at cristianleo120@gmail.com.

---

Happy visualizing! 🎨📈