import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

path_dir = os.path.dirname(os.path.abspath(__file__))

# Load and process data
with open(path_dir+'/input.txt', 'r') as f:
    data = f.read()

chars = list(set(data))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

vocab_size = len(chars)
hidden_size = 100
seq_length = 25
learning_rate = 0.1

# Convert text to numerical indices
def encode_text(text):
    return [char_to_ix[ch] for ch in text]

def decode_text(indices):
    return ''.join(ix_to_char[i] for i in indices)

# Define PyTorch RNN model
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(vocab_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden

# Initialize model, loss function, optimizer
model = CharRNN(vocab_size, hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Function to generate text
def sample(model, start_char, length=200):
    model.eval()
    input_char = torch.zeros(1, 1, vocab_size)
    input_char[0, 0, char_to_ix[start_char]] = 1

    hidden = torch.zeros(1, 1, hidden_size)
    output_text = start_char

    for _ in range(length):
        output, hidden = model(input_char, hidden)
        probs = torch.softmax(output[0, 0], dim=0).detach().numpy()
        next_char = np.random.choice(range(vocab_size), p=probs)
        output_text += ix_to_char[next_char]
        input_char = torch.zeros(1, 1, vocab_size)
        input_char[0, 0, next_char] = 1

    return output_text

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    hprev = torch.zeros(1, 1, hidden_size)

    for i in range(0, len(data) - seq_length, seq_length):
        inputs = encode_text(data[i:i+seq_length])
        targets = encode_text(data[i+1:i+seq_length+1])

        input_tensor = torch.zeros(1, seq_length, vocab_size)
        target_tensor = torch.tensor(targets, dtype=torch.long).unsqueeze(0)

        for t, char in enumerate(inputs):
            input_tensor[0, t, char] = 1

        optimizer.zero_grad()
        output, _ = model(input_tensor, hprev)
        loss = criterion(output.view(-1, vocab_size), target_tensor.view(-1))
        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
        print(sample(model, start_char='H'))

