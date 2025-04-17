import torch
import torch.nn as nn
from torch.autograd import Variable
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Neural Network Architecture
        self.hidden_layer1 = nn.Linear(2,5)
        self.hidden_layer2 = nn.Linear(5,5)
        self.hidden_layer3 = nn.Linear(5,5)
        self.hidden_layer4 = nn.Linear(5,5)
        self.hidden_layer5 = nn.Linear(5,5)
        self.output_layer = nn.Linear(5,1)

    def forward(self, x,t):
        # Forward pass through the network
        inputs = torch.cat([x,t],axis=1)  # Combine x and t to form input vector
        layer1_out = torch.sigmoid(self.hidden_layer1(inputs))
        layer2_out = torch.sigmoid(self.hidden_layer2(layer1_out))
        layer3_out = torch.sigmoid(self.hidden_layer3(layer2_out))
        layer4_out = torch.sigmoid(self.hidden_layer4(layer3_out))
        layer5_out = torch.sigmoid(self.hidden_layer5(layer4_out))
        output = self.output_layer(layer5_out)  # Output for regression, no activation
        return output

### (2) Model Initialization
net = Net()
net = net.to(device)
mse_cost_function = torch.nn.MSELoss()  # Mean squared error loss function
optimizer = torch.optim.Adam(net.parameters())  # Adam optimizer

## PDE as loss function. Thus would use the network which we call as u_theta
def f(x,t, net, alpha):
    x.requires_grad_(True)  # Enable gradient tracking for x
    t.requires_grad_(True)  # Enable gradient tracking for t
    u = net(x,t)  # Network output (dependent variable u based on independent variables x,t)
    
    # Compute the gradient of u with respect to x and t (du/dx, du/dt)
    du_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    du_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True, retain_graph=True)[0]

    # Second derivative of u with respect to x (du/dx^2)
    ddu_x = torch.autograd.grad(du_x, x, torch.ones_like(du_x), create_graph=True, retain_graph=True)[0]
    
    pde = du_t - alpha * ddu_x
    return pde




### (3) Training / Fitting
iterations = 20000  # Number of training iterations
previous_validation_loss = 99999999.0  # Initial large validation loss for comparison
for epoch in range(iterations):
    optimizer.zero_grad()  # Zero out gradients at the beginning of each epoch

    # Define Boundary Conditions (BC)
    N_bc = 100
    t_bc_0 = torch.linspace(0,1,N_bc).view(-1,1).requires_grad_(True).to(device)
    t_bc_1 = torch.linspace(0,1,N_bc).view(-1,1).requires_grad_(True).to(device)
    x_bc_0 = torch.zeros(N_bc,1).requires_grad_(True).to(device)  # x=0
    x_bc_1 = torch.ones(N_bc,1).requires_grad_(True).to(device)  # x=1
    alpha = 0.01

    # Set the boundary condition for u(x,t) (u_bc)
    u_bc_0 = torch.zeros(N_bc,1).to(device)  # u(x=0,t) = 0
    u_bc_1 = torch.zeros(N_bc,1).to(device)  # u(x=1,t) = 0

    # Set initial conditions (u(x,0))
    N_ic = 100
    x_ic = torch.linspace(0,1,N_ic).view(-1,1).requires_grad_(True).to(device)  # x values
    t_ic = torch.zeros(N_ic,1).requires_grad_(True).to(device)  # t=0

    u_ic = torch.sin(np.pi*x_ic)

    # TO DO: Loss based on boundary conditions (use BC dataset)
    # pt_x_bc = None  # TO DO: Convert x_bc to torch variable
    # pt_t_bc = None  # TO DO: Convert t_bc to torch variable
    # pt_u_bc = None  # TO DO: Convert u_bc to torch variable

    net_bc_0_out = net(x_bc_0, t_bc_0)  # Output from the network for boundary conditions
    net_bc_1_out = net(x_bc_1, t_bc_1)  # Output from the network for boundary conditions
    net_ic_out = net(x_ic, t_ic)

    mse_ubc_0 = mse_cost_function(net_bc_0_out, u_bc_0)  # BC loss (MSE between predicted and actual u)
    mse_ubc_1 = mse_cost_function(net_bc_1_out, u_bc_1)  # BC loss (MSE between predicted and actual u)
    mse_uic = mse_cost_function(net_ic_out, u_ic)  # IC loss (MSE between predicted and actual u)

    # TO DO: Collocation points for PDE loss (random points in the domain where the PDE should be satisfied)
    x_collocation = torch.linspace(0,1,100).view(-1,1).requires_grad_(True).to(device)
    t_collocation = torch.linspace(0,1,100).view(-1,1).requires_grad_(True).to(device)
    all_zeros = torch.zeros(x_collocation.shape[0],1).requires_grad_(True).to(device)  # Placeholder for PDE residual

    # TO DO: Compute the PDE residual using the function f(x,t)
    f_out = f(x_collocation, t_collocation, net, alpha)  # PDE residual
    mse_f = mse_cost_function(f_out, all_zeros)  # PDE loss (MSE between predicted and actual PDE residual)

    # Combining the boundary condition loss and PDE loss
    loss = mse_f + mse_ubc_0 + mse_ubc_1 + mse_uic  # Total loss

    # Backpropagation and optimizer step
    loss.backward()  # Backpropagate the loss
    optimizer.step()  # Update model parameters based on gradients

    # TO DO: Optional validation step (for testing during training)
    with torch.autograd.no_grad():
        if epoch % 100 == 0:  # Print every 100 epochs
            print(epoch, "Training Loss:", loss.item())

# Visualization (Optional) - Plotting the solution to visualize progress
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

x=np.arange(0,2,0.02)  # Define the range for x
t=np.arange(0,1,0.02)  # Define the range for t
ms_x, ms_t = np.meshgrid(x, t)  # Create meshgrid for plotting
x = np.ravel(ms_x).reshape(-1,1)  # Flatten the meshgrid for input
t = np.ravel(ms_t).reshape(-1,1)  # Flatten the meshgrid for input

pt_x = torch.tensor(x, dtype=torch.float32)  # Convert x to torch variable
pt_t = torch.tensor(t, dtype=torch.float32)  # Convert t to torch variable
pt_u = net(pt_x, pt_t)  # Get the prediction from the network
pt_u = pt_u.detach()  # Detach from the graph

u = pt_u.data.cpu().numpy()  # Convert the prediction back to numpy array
ms_u = u.reshape(ms_x.shape)  # Reshape for plotting

# Plotting the solution surface
surf = ax.plot_surface(ms_x, ms_t, ms_u, cmap=cm.coolwarm, linewidth=0, antialiased=False)
             
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig('pinnOut.jpg')  # Save the plot as an image
plt.show()  # Display the plot
