import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# Define the neural network architecture
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.hidden_layers.append(nn.Linear(layers[i], layers[i+1]))
    
    def forward(self, x, t):
        z = torch.cat([x, t], dim=1)  # Concatenate space and time
        for layer in self.hidden_layers:
            z = torch.tanh(layer(z))
        return z

# Define the 1D heat equation (PDE residual)
def heat_equation(u, x, t, model, alpha=0.01):
    # TO DO: Compute the first derivative of u with respect to t (u_t)
    u_t = None  # Replace with the correct derivative

    # TO DO: Compute the second derivative of u with respect to x (u_xx)
    u_xx = None  # Replace with the correct derivative
    
    return u_t - alpha * u_xx

# Initial condition
def initial_condition(x):
    # TO DO: Set the initial condition at t=0
    return torch.sin(np.pi * x)  # replace with correct form as needed

# Boundary condition at x=0 and x=1
def boundary_condition(x, t):
    # TO DO: Set the boundary condition values at x=0 and x=1
    return torch.zeros_like(x), torch.zeros_like(t)  #  adjust as needed

# Training setup
def train(model, num_epochs=10000, learning_rate=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_history = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()

        # Sample points from the domain
        x = torch.rand(1000, 1).requires_grad_(True)  # Random x in [0, 1]
        t = torch.rand(1000, 1).requires_grad_(True)  # Random t in [0, T]
        
        # Evaluate model at x, t
        u = model(x, t)

        # TO DO: Compute PDE residual loss (loss_pde) from the PDE equation
        loss_pde = None  # Replace with the correct loss computation

        # Initial condition loss (at t = 0)
        x_ic = torch.rand(100, 1).requires_grad_(True)  # Random x values in [0, 1]
        u_ic = model(x_ic, torch.zeros_like(x_ic))  # IC is at t=0
        # TO DO: Compute the initial condition loss (loss_ic)
        loss_ic = None  # Replace with correct loss computation

        # Boundary condition loss (u = 0 at x=0 and x=1, for all t)
        batch_size = 100  # Ensure consistent batch size
        x_bc_0 = torch.zeros(batch_size, 1).requires_grad_(True)  # Boundary condition at x=0
        x_bc_1 = torch.ones(batch_size, 1).requires_grad_(True)  # Boundary condition at x=1
        t_bc = torch.rand(batch_size, 1).requires_grad_(True)  # Random time steps for boundary conditions

        # TO DO: Apply boundary condition and compute loss
        u_bc_0 = model(x_bc_0, t_bc)
        u_bc_1 = model(x_bc_1, t_bc)
        # TO DO: Compute the boundary condition loss (loss_bc)
        loss_bc = None  # Replace with correct loss computation

        # Total loss
        loss = loss_pde + loss_ic + loss_bc

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item()}")

    return model, loss_history

# Define the network and layers (input: x and t, output: u)
layers = [2, 20, 20, 20, 1]  # Example architecture
model = PINN(layers)

# Train the model
model, loss_history = train(model)

# Plot the loss history
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Visualize solution at t = 1
x_plot = torch.linspace(0, 1, 100).reshape(-1, 1)
t_plot = torch.ones_like(x_plot)  # Example: plot at t=1
u_plot = model(x_plot, t_plot)

plt.plot(x_plot.detach().numpy(), u_plot.detach().numpy(), label='Predicted u(x, t=1)')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.title('Predicted Solution at t=1')
plt.legend()
plt.show()
