import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F


# Define transformations: Convert images to tensor and normalize
transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
# Download and load the training and test datasets
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transformer)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformer)
# Load data into DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3) # Conv layer 1: 1 input, 32 output, kernel size 3
        self.conv2 = nn.Conv2d(32, 64, 3) # Conv layer 2: 32 input, 64 output, kernel size 3
        self.fc1 = nn.Linear(64 * 5 * 5, 128) # Fully connected: 64 * 5 * 5 input size, 128 output
        self.fc2 = nn.Linear(128, 10) # Output layer: 128 input, 10 classes (digits)

    def forward(self, x):
        x = F.relu(self.conv1(x)) # Apply relu activation after conv1 # out: 26 x 26
        x = F.max_pool2d(x, 2, 2) # Apply max pooling # out: 13 x 13
        x = F.relu(self.conv2(x)) # Apply relu activation after conv2 # out: 11 x 11
        x = F.max_pool2d(x, 2, 2) # Apply max pooling # out: 5 x 5
        x = x.view(-1, 64 * 5 * 5) # Flatten the output
        x = F.relu(self.fc1(x)) # Apply relu activation after fc1
        x = self.fc2(x) # Output layer
        return x
    
import torch.optim as optim
# Initialize model, loss function, and optimizer
model = CNNModel()
criterion = nn.CrossEntropyLoss() # Cross entropy loss for multi-class classification
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training the model
num_epochs = 5
for epoch in range(num_epochs):
    model.train() # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(train_loader, 0):
        optimizer.zero_grad() # Zero the gradients before the backward pass
        outputs = model(inputs) # Forward pass
        loss = criterion(outputs, labels) # Compute loss
        loss.backward() # Backward pass (compute gradients)
        optimizer.step() # Update weights using the optimizer
        _, predicted = torch.max(outputs.data, 1) # Get predictions
        total += labels.size(0) # Total number of samples
        correct += (predicted == labels).sum().item() # Count correct predictions
        running_loss += loss.item() # Add the loss to running total

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, \
    Accuracy: {100 * correct / total:.2f}%")

# Testing the model
model.eval() # Set model to evaluation mode
correct = 0
total = 0
with torch.no_grad(): # No gradients needed for evaluation
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1) # Get predictions
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")