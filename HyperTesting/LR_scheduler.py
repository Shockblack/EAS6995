import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Define the CNN for FashionMNIST
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)

# Load FashionMNIST
def get_fashion_mnist(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

# Noam Scheduler (from "Attention is All You Need")
class NoamOpt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        self._step += 1
        rate = self.rate()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step=None):
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** -0.5) * min(step ** -0.5, step * self.warmup ** -1.5)

# Training loop
def train_model(scheduler_type='noam', epochs=20, model_size=512, factor=1, warmup=4000, eta_max=0.1):
    trainloader, testloader = get_fashion_mnist()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)

    # Set an initial learning rate for the optimizer (important for CosineAnnealingLR)
    base_optimizer = optim.Adam(model.parameters(), lr=0.001)  # Set initial learning rate here

    if scheduler_type == 'noam':
        scheduler = NoamOpt(model_size=model_size, factor=factor, warmup=warmup, optimizer=base_optimizer)
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base_optimizer, T_max=epochs * len(trainloader), eta_min=0)
    else:
        raise ValueError("Unsupported scheduler type.")

    train_loss, val_loss, train_acc, val_acc, lr_values = [], [], [], [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct, total = 0, 0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            base_optimizer.zero_grad()  # Zero gradients before the backward pass
            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            
            if scheduler_type == 'noam':
                scheduler.step()  # Step the scheduler for Noam
                lr_values.append(scheduler._rate)
            else:
                base_optimizer.step()  # Step the optimizer for Cosine Annealing
                scheduler.step()  # Step the scheduler for Cosine Annealing
                lr_values.append(base_optimizer.param_groups[0]['lr'])  # Get the learning rate

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss.append(total_loss / len(trainloader))
        train_acc.append(100 * correct / total)

        # Validation
        model.eval()
        val_correct, val_total, val_running_loss = 0, 0, 0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss.append(val_running_loss / len(testloader))
        val_acc.append(100 * val_correct / val_total)

        print(f"[{scheduler_type.upper()}] Epoch {epoch+1}/{epochs} - "
              f"Train Acc: {train_acc[-1]:.2f}% | Val Acc: {val_acc[-1]:.2f}% | LR: {lr_values[-1]:.5f}")

    return train_loss, val_loss, train_acc, val_acc, lr_values


# Run and Plot
if __name__ == '__main__':
    noam_results = train_model(scheduler_type='noam', epochs=20)
    cosine_results = train_model(scheduler_type='cosine', epochs=20, eta_max=0.01)

    def plot_results(train_loss, val_loss, train_acc, val_acc, lr_values, title_suffix):
        fig, ax = plt.subplots(3, 1, figsize=(10, 14))

        ax[0].plot(train_acc, label="Train Acc")
        ax[0].plot(val_acc, label="Val Acc")
        ax[0].set_title(f"{title_suffix}: Accuracy")
        ax[0].legend()

        ax[1].plot(train_loss, label="Train Loss")
        ax[1].plot(val_loss, label="Val Loss")
        ax[1].set_title(f"{title_suffix}: Loss")
        ax[1].legend()

        ax[2].plot(lr_values)
        ax[2].set_title(f"{title_suffix}: Learning Rate Over Steps")

        plt.tight_layout()
        plt.savefig(f'{title_suffix.lower()}_training_plot.jpg')
        plt.show()

        # Save for later analysis
        with open(f'{title_suffix.lower()}_training_loss.pkl', 'wb') as f:
            pickle.dump(train_loss, f)
        with open(f'{title_suffix.lower()}_val_loss.pkl', 'wb') as f:
            pickle.dump(val_loss, f)

    plot_results(*noam_results, title_suffix='Noam')
    plot_results(*cosine_results, title_suffix='CosineAnnealing')
