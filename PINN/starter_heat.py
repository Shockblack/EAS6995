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
def f(x,t, net):
    u = net(x,t)  # Network output (dependent variable u based on independent variables x,t)
    
    # TO DO: Compute the gradient of u with respect to x and t (du/dx, du/dt)
    u_x = None  # TO DO: Compute du/dx
    u_t = None  # TO DO: Compute du/dt
    
    pde = u_x - 2*u_t - u  # PDE residual (replace with your own equation if needed)
    return pde

# TO DO: Define Boundary Conditions (BC)
x_bc = None  # TO DO: Define boundary condition x values
t_bc = None  # TO DO: Define boundary condition t values

# TO DO: Set the boundary condition for u(x,t) (u_bc)
u_bc = None  # TO DO: Define boundary condition for u(x,t)

### (3) Training / Fitting
iterations = 20000  # Number of training iterations
previous_validation_loss = 99999999.0  # Initial large validation loss for comparison
for epoch in range(iterations):
    optimizer.zero_grad()  # Zero out gradients at the beginning of each epoch

    # TO DO: Loss based on boundary conditions (use BC dataset)
    pt_x_bc = None  # TO DO: Convert x_bc to torch variable
    pt_t_bc = None  # TO DO: Convert t_bc to torch variable
    pt_u_bc = None  # TO DO: Convert u_bc to torch variable

    net_bc_out = net(pt_x_bc, pt_t_bc)  # Output from the network for boundary conditions
    mse_u = mse_cost_function(net_bc_out, pt_u_bc)  # BC loss (MSE between predicted and actual u)

    # TO DO: Collocation points for PDE loss (random points in the domain where the PDE should be satisfied)
    x_collocation = None  # TO DO: Define collocation x points
    t_collocation = None  # TO DO: Define collocation t points
    all_zeros = None  # TO DO: Define zeros for PDE residual loss

    pt_x_collocation = None  # TO DO: Convert collocation x points to torch variable
    pt_t_collocation = None  # TO DO: Convert collocation t points to torch variable
    pt_all_zeros = None  # TO DO: Convert zeros to torch variable

    # TO DO: Compute the PDE residual using the function f(x,t)
    f_out = None  # TO DO: Call f(pt_x_collocation, pt_t_collocation, net)
    mse_f = None  # TO DO: Compute PDE loss

    # Combining the boundary condition loss and PDE loss
    loss = mse_u + mse_f

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

pt_x = None  # TO DO: Convert x to torch variable
pt_t = None  # TO DO: Convert t to torch variable
pt_u = None  # TO DO: Get network prediction

u = pt_u.data.cpu().numpy()  # Convert the prediction back to numpy array
ms_u = u.reshape(ms_x.shape)  # Reshape for plotting

# Plotting the solution surface
surf = ax.plot_surface(ms_x, ms_t, ms_u, cmap=cm.coolwarm, linewidth=0, antialiased=False)
             
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig('pinnOut.jpg')  # Save the plot as an image
plt.show()  # Display the plot
