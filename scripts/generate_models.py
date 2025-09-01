import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

DATA_FOLDER = "data"

def generate_torch_model():
    # ----- 1. Create dummy data -----
    # y = 2x + 1
    X = torch.linspace(-1, 1, 100).reshape(-1, 1)
    y = 2 * X + 1 + 0.1 * torch.randn(X.shape)

    # ----- 2. Define model -----
    class LinearRegression(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1)

        def forward(self, x):
            return self.linear(x)

    model = LinearRegression()

    # ----- 3. Train -----
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(200):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

    print("Trained weights:", model.linear.weight.item())
    print("Trained bias:", model.linear.bias.item())

    # ----- 4. Export to ONNX -----
    dummy_input = torch.randn(1, 1)
    torch.onnx.export(
        model,
        dummy_input,
        f"{DATA_FOLDER}/torch_linear_regression.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11,
        dynamo=False
    )

    print("✅ Model exported to linear_regression.onnx")

def generate_sklearn_model():
    # save as train_export.py

    # --- generate sample tabular data ---
    X, y = make_regression(n_samples=10, n_features=3, noise=0.1, random_state=42)

    # --- train a simple model ---
    model = LinearRegression()
    model.fit(X, y)

    # --- convert to ONNX ---
    initial_type = [('input', FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    # --- save ONNX model ---
    with open(f"{DATA_FOLDER}/sk_linear_regression.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())

    print("ONNX model saved as linear_regression.onnx")

def main():
    generate_torch_model()
    generate_sklearn_model()

if __name__ == "__main__":
    main()