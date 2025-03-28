# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and process data
with open('input.txt', 'r') as f:
    data = f.read()

chars = list(set(data))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

vocab_size = len(chars)
hidden_size = 100
seq_length = 25
num_layers = 2  # Number of LSTM layers
dropout_rate = 0.3
learning_rate = 0.1

# Convert text to numerical indices
def encode_text(text):
    return [char_to_ix[ch] for ch in text]

def decode_text(indices):
    return ''.join(ix_to_char[i] for i in indices)

# Define PyTorch LSTM model
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=1, dropout_rate=0.3):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(vocab_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        """Initialize hidden state (h, c) for LSTM"""
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device))

# Initialize model, loss function, optimizer
model = CharLSTM(vocab_size, hidden_size, num_layers, dropout_rate).to(device)  # Move model to GPU
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Function to generate text
def sample(model, start_char, length=200):
    model.eval()
    input_char = torch.zeros(1, 1, vocab_size, device=device)
    input_char[0, 0, char_to_ix[start_char]] = 1

    hidden = model.init_hidden(batch_size=1)
    output_text = start_char

    for _ in range(length):
        output, hidden = model(input_char, hidden)
        probs = torch.softmax(output[0, 0], dim=0).detach().cpu().numpy()
        next_char = np.random.choice(range(vocab_size), p=probs)
        output_text += ix_to_char[next_char]
        input_char = torch.zeros(1, 1, vocab_size, device=device)
        input_char[0, 0, next_char] = 1

    return output_text

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    hidden = model.init_hidden(batch_size=1)  # Initialize LSTM hidden state on GPU

    for i in range(0, len(data) - seq_length, seq_length):
        inputs = encode_text(data[i:i+seq_length])
        targets = encode_text(data[i+1:i+seq_length+1])

        # Move tensors to GPU
        input_tensor = torch.zeros(1, seq_length, vocab_size, device=device)
        target_tensor = torch.tensor(targets, dtype=torch.long, device=device).unsqueeze(0)

        for t, char in enumerate(inputs):
            input_tensor[0, t, char] = 1

        optimizer.zero_grad()
        hidden = tuple(h.detach() for h in hidden)  # Detach to prevent gradient accumulation

        output, hidden = model(input_tensor, hidden)
        loss = criterion(output.view(-1, vocab_size), target_tensor.view(-1))
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
        print(sample(model, start_char='H'))
