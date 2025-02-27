import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm


# Load the FashionMNIST dataset 
def load_data(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()])  # Only convert to tensor

    # Download the dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # --- Normalize data manually to the range [0, 1] ---
    # Normalize to [0, 1]
    # Data has a range of [0, 255]
    # train_dataset.data = train_dataset.data / 255
    # test_dataset.data = test_dataset.data / 255

    # Subsampling: 50% from each class
    train_indices = subsample_50_percent_per_class(train_dataset)
    train_subset = Subset(train_dataset, train_indices)

    # DataLoader for batching
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

# Function to perform subsampling 50% from each class
def subsample_50_percent_per_class(dataset):
    """
    Subsample 50% of the data from each class.
    dataset: The full dataset (e.g., FashionMNIST)
    Returns: A list of indices for the subsampled dataset
    """
    # --- Implement subsampling logic here ---
    sampled_indices = np.arange(len(dataset))

    # Shuffle the indices
    np.random.shuffle(sampled_indices)
    sampled_indices = sampled_indices[:len(sampled_indices) // 2]

    return sampled_indices


# Forward pass for Fully Connected Layer
def fully_connected_forward(X, W, b):
    """
    Perform forward pass for a fully connected (linear) layer.
    X: Input data
    W: Weight matrix
    b: Bias vector
    """
    Z = X @ W + b # Compute linear output
    return Z

# Forward pass for ReLU activation
def relu_forward(Z):
    """
    ReLU activation function forward pass.
    Z: Linear output (input to ReLU)
    """
    # Want Z if Z > 0, 0 otherwise
    A = np.maximum(0, Z)
    return A

# Forward pass for Softmax activation
def softmax_forward(Z):
    """
    Softmax activation function forward pass.
    Z: Output logits (before softmax)
    """
    exp_z = np.exp(Z - np.max(Z, axis=1, keepdims=True))  # Apply softmax function
    output = exp_z/np.sum(exp_z, axis=1, keepdims=True)  # Normalize exp_z to get the softmax output
    return output

# Backward pass for Fully Connected Layer (Linear)
def fully_connected_backward(X, Z, W, dZ):
    """
    Compute gradients for the fully connected (linear) layer.
    X: Input data
    Z: Output of the layer before activation (logits)
    W: Weight matrix
    dZ: Gradient of the loss with respect to Z (from the next layer)
    """
    dW = X.T @ dZ / X.shape[0]  # Compute gradient of loss with respect to weights
    db = np.sum(dZ, axis=0) / X.shape[0]  # Compute gradient of loss with respect to biases
    dZ = dZ @ W.T  # Compute gradient of loss with respect to Z
    return dW, db, dZ

# Backward pass for ReLU activation
def relu_backward(Z, dA):
    """
    Compute the gradient for ReLU activation.
    Z: Input to ReLU (before activation)
    dA: Gradient of the loss with respect to activations (from the next layer)
    """
    dZ = (Z > 0).astype(float)*dA  # Derivative of ReLU: 1 if Z > 0, 0 otherwise
    return dZ

# Backward pass for Softmax Layer
def softmax_backward(S, Y):
    """
    Compute the gradient of the loss with respect to softmax output.
    S: Output of softmax
    Y: True labels (one-hot encoded)
    """
    dZ = S - Y
    return dZ

# Weight update function (gradient descent)
def update_weights(weights, biases, grads_W, grads_b, learning_rate=0.01):
    """
    --- TODO: Implement the weight update step ---
    weights: Current weights
    biases: Current biases
    grads_W: Gradient of the weights
    grads_b: Gradient of the biases
    learning_rate: Learning rate for gradient descent
    """
    weights_updated = weights - learning_rate * grads_W
    biases_updated = biases - learning_rate * grads_b
    return weights_updated, biases_updated


# Define the neural network 
def train(train_loader, test_loader, epochs=10000, learning_rate=0.01):
    # Initialize weights and biases
    input_dim = train_loader.dataset[0][0].shape[1] * train_loader.dataset[0][0].shape[2]
    hidden_dim1 = 128   #could set differently
    hidden_dim2 = 64    #could set differently
    output_dim = 10
    
    # Initialize weights randomly
    print("Initializing weights...")
    W1 = np.random.randn(input_dim, hidden_dim1) * 0.01
    b1 = np.zeros(hidden_dim1)
    W2 = np.random.randn(hidden_dim1, hidden_dim2) * 0.01
    b2 = np.zeros(hidden_dim2)
    W3 = np.random.randn(hidden_dim2, output_dim) * 0.01
    b3 = np.zeros(output_dim)
    
    training_loss = []
    test_loss = []
    training_accuracy = []
    test_accuracy = []

    # Loop through epochs
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        test_epoch_loss = 0
        correct_predictions_total = 0
        total_samples = 0

        for batch_idx, (X_batch, Y_batch) in enumerate(train_loader):
            # Flatten images to vectors
            X_batch = X_batch.numpy().reshape(X_batch.shape[0], -1)
            Y_batch = torch.eye(output_dim)[Y_batch]  # Map label indices to corresponding one-hot encoded vectors
            Y_batch = Y_batch.numpy()

            # --- Implement the forward pass ---
            Z1 = fully_connected_forward(X_batch, W1, b1)
            A1 = relu_forward(Z1)
            Z2 = fully_connected_forward(A1, W2, b2)
            A2 = relu_forward(Z2)
            Z3 = fully_connected_forward(A2, W3, b3)
            Y_pred = softmax_forward(Z3)
            
            # --- Implement loss computation ---
            # Cross-entropy loss
            loss = - np.sum(Y_batch * np.log(Y_pred + 1e-8)) / Y_batch.shape[0]

            epoch_loss = epoch_loss + loss

            # --- Implement backward pass ---
            dZ3 = softmax_backward(Y_pred, Y_batch)
            dW3, db3, dA2 = fully_connected_backward(A2, Z3, W3, dZ3)
            dZ2 = relu_backward(Z2, dA2)
            dW2, db2, dA1 = fully_connected_backward(A1, Z2, W2, dZ2)
            dZ1 = relu_backward(Z1, dA1)
            dW1, db1, _ = fully_connected_backward(X_batch, Z1, W1, dZ1)


            # --- Implement weight update ---
            W1, b1 = update_weights(W1, b1, dW1, db1, learning_rate)
            W2, b2 = update_weights(W2, b2, dW2, db2, learning_rate)
            W3, b3 = update_weights(W3, b3, dW3, db3, learning_rate)



            # Track accuracy
            Y_pred = np.argmax(Y_pred, axis=1)
            Y_batch = np.argmax(Y_batch, axis=1)
            correct_predictions = Y_pred == Y_batch
            correct_predictions_total = correct_predictions_total + np.sum(correct_predictions)
            total_samples = total_samples + X_batch.shape[0]
            

        # Print out the progress
        train_accuracy = correct_predictions_total / total_samples
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader)}, Accuracy: {train_accuracy * 100}%")

        training_loss.append(epoch_loss / len(train_loader))
        training_accuracy.append(train_accuracy)

        # For every 100 epochs, get the validation loss and error
        if (epoch + 1) % 10 == 0:
            test_X, test_Y = next(iter(test_loader))
            test_X = test_X.numpy().reshape(test_X.shape[0], -1)
            test_Y = torch.eye(output_dim)[test_Y]
            test_Y = test_Y.numpy()
            Z1 = fully_connected_forward(test_X, W1, b1)
            A1 = relu_forward(Z1)
            Z2 = fully_connected_forward(A1, W2, b2)
            A2 = relu_forward(Z2)
            Z3 = fully_connected_forward(A2, W3, b3)
            Y_pred = softmax_forward(Z3)

            loss = - np.sum(test_Y * np.log(Y_pred))
            correct_predictions = np.argmax(Y_pred, axis=1) == np.argmax(test_Y, axis=1)
            accuracy = np.sum(correct_predictions) / test_Y.shape[0]
            print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy * 100}%")
            test_loss.append(loss)
            test_accuracy.append(accuracy)

        # Early stopping: Stop training if the relative loss improvement is less than 1e-3 for 10 epochs
        if len(training_loss) > 10 and all(np.abs(np.diff(training_loss[-11:]) / training_loss[-11:-1]) < 1e-3):
            print("Early stopping: Loss has not decreased for 10 epochs.")
            break

        
    print("Training complete!")
    return training_loss, test_loss, training_accuracy, test_accuracy

# Main function
def main():
    batch_size = 64
    train_loader, test_loader = load_data(batch_size)
    epochs = 100
    # Start training
    training_loss, test_loss, training_accuracy, test_accuracy = \
        train(train_loader, test_loader, epochs=epochs, learning_rate=0.1)
    
    # PLOT TRAINING LOSS AND TEST LOSS ON ONE SUBPLOT (epoch vs loss)
# PLOT TRAINING ACCURACY AND TEST ACCURACY ON A SECOND SUBPLOT (epoch vs accuracy)

    epochs_train = list(range(1, len(training_loss) + 1)) # Epochs for training loss (1, 2, ..., N)

    epochs_test = list(range(10, (len(test_loss) + 1) * 10, 10)) # Epochs for test loss (100, 200, ..., N*100)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Training and Test Loss on the first subplot
    ax1.plot(epochs_train, training_loss, label='Training Loss', color='blue', marker='o')
    ax1.plot(epochs_test, test_loss, label='Test Loss', color='red', marker='x')
    ax1.set_title('Loss vs Epoch')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot Training and Test Accuracy on the second subplot
    ax2.plot(epochs_train, training_accuracy, label='Training Accuracy', color='blue', marker='o')
    ax2.plot(epochs_test, test_accuracy, label='Test Accuracy', color='red', marker='x')
    ax2.set_title('Accuracy vs Epoch')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
