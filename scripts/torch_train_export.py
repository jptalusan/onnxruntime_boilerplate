import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch.onnx
import json
# For use with the infer.cpp that has loadFeatureMapping().

DATA_FOLDER = "data"

# --- 1. Generate Dummy Data ---
# A simple DataFrame with one categorical and one numerical feature.
data = {
    'city': ['New York', 'London', 'Paris', 'New York', 'London', 'Paris', 'New York', 'London', 'Paris'],
    'age': [30, 25, 40, 35, 28, 50, 45, 33, 22],
    'target': [1, 0, 1, 1, 0, 1, 0, 1, 0] # Binary classification target
}
df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)
print("-" * 30)

# --- 2. Preprocess Data (One-Hot Encoding) ---
# Create an instance of the OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# Fit and transform the categorical 'city' column
cat_features_encoded = encoder.fit_transform(df[['city']])

# Create a DataFrame with the one-hot encoded features
cat_df = pd.DataFrame(cat_features_encoded, columns=encoder.get_feature_names_out(['city']))

# Combine the one-hot encoded features with the numerical 'age' feature
X_df = pd.concat([cat_df, df[['age']]], axis=1)

# Get the number of input features for the model
input_features = X_df.shape[1]

# Convert the DataFrame to PyTorch tensors
X = torch.tensor(X_df.values, dtype=torch.float32)
y = torch.tensor(df['target'].values, dtype=torch.float32).view(-1, 1)

print("Preprocessed Features (X) and Target (y):")
print(X)
print(y)
print("-" * 30)

# --- 3. Save the Feature Mapping to a JSON file ---
# Extract the feature names from the one-hot encoder
feature_names = list(cat_df.columns)
# Add the numerical feature name to maintain order
feature_names.append('age')

mapping = {
    "features": feature_names,
    "categorical_mapping": encoder.get_feature_names_out(['city']).tolist()
}

with open(f"{DATA_FOLDER}/features_mapping.json", "w") as f:
    json.dump(mapping, f, indent=4)

print("Feature mapping successfully saved to features_mapping.json")

# --- 4. Define the PyTorch Model ---
class SimpleNN(nn.Module):
    def __init__(self, input_features):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_features, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# Instantiate the model
model = SimpleNN(input_features)

# --- 5. Train the Model ---
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("\nTraining complete.")

# --- 6. Export to ONNX ---
# Create a dummy input tensor with the same shape as the training data
dummy_input = torch.randn(1, input_features)

# Export the model
try:
    torch.onnx.export(
        model,
        dummy_input,
        f"{DATA_FOLDER}/model.onnx",
        input_names=['input'],
        output_names=['output'],
        verbose=False,
        opset_version=14 # Recommended for compatibility
    )
    print("\nModel successfully exported to model.onnx")
except Exception as e:
    print(f"\nError during ONNX export: {e}")
