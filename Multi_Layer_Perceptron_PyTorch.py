import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super(MultiLayerPerceptron, self).__init__()
        self.model = nn.Sequential(nn.Linear(64, 100), nn.ReLU(), nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 10))
    def forward(self, x):
        return self.model(x)

mlp = MultiLayerPerceptron()
digits = load_digits()
x, y = digits.data, digits.target

scaler = StandardScaler()
x = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)

epochs = 50
for i in range(epochs):
    optimizer.zero_grad()
    outputs = mlp(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (i + 1) % 5 == 0:
        print(f"Epoch {i + 1}/{epochs}, Loss: {loss.item():.4f}")

with torch.no_grad():
    test_outputs = mlp(x_test_tensor)
    _, predictions = torch.max(test_outputs, 1)
    accuracy = accuracy_score(y_test_tensor, predictions)
    print(f"Test Accuracy: {accuracy:.2f}")
    print("Classification Report\n", classification_report(y_test, predictions))
    torch.save(mlp.state_dict(), "Models and Datasets/multilayer_perceptron_pytorch.pth")

print("Model has been successfully saved!")