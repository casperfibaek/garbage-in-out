import os
import numpy as np
import torch
import torch.nn as nn
import pdb

from model_arch import NetworkNorm, NetworkTotal


torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)

DATA_FOLDER = "./data"
MODEL_FOLDER = "./models"
DATA_X = np.load(os.path.join(DATA_FOLDER, "x_train.npy"))
DATA_Y = np.load(os.path.join(DATA_FOLDER, "y_train.npy"))

# shuffle the data
indices = np.arange(len(DATA_X))
np.random.shuffle(indices)
DATA_X = DATA_X[indices]
DATA_Y = DATA_Y[indices]

# (DATA_X.sum(axis=1) != DATA_Y.sum(axis=1)).sum() # how many outputs are not equal to inputs
# ((DATA_Y.sum(axis=1) / DATA_X.sum(axis=1)) > 1).sum() # how many outputs are greater than inputs

y_total = (DATA_Y.sum(axis=1) / DATA_X.sum(axis=1))[..., None]
y_norm = DATA_Y / DATA_Y.sum(axis=1)[:, None]
DATA_Y = np.concatenate([y_total, y_norm], axis=1)

# split the data into training and validation
split = int(len(DATA_X) * 0.8)
x_train, x_val = DATA_X[:split], DATA_X[split:]
y_train, y_val = DATA_Y[:split], DATA_Y[split:]

# convert the data to tensors
x_train = torch.tensor(x_train, dtype=torch.float32)
x_val = torch.tensor(x_val, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

x_indices = x_train.shape[1]  # 489
y_indices = y_train.shape[1]  # 434

SIZE = 512
DROPOUT = 0.2


# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_train = x_train.to(device)
x_val = x_val.to(device)
y_train = y_train.to(device)
y_val = y_val.to(device)

# train the model
model_norm = NetworkNorm(x_indices, y_indices, SIZE=SIZE, DROPOUT=DROPOUT)
optimizer = torch.optim.AdamW(model_norm.parameters(), lr=0.0001, weight_decay=0.01)
loss_fn = nn.CrossEntropyLoss()

model_norm.to(device)

# Training loop
epochs = 50
batch_size = 32

for epoch in range(epochs):
    model_norm.train()
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i : i + batch_size]
        y_batch = y_train[i : i + batch_size][:, 1:]

        optimizer.zero_grad()
        output = model_norm(x_batch)
        loss = loss_fn(output, y_batch)
        loss.backward()
        optimizer.step()

    # Validation loss
    model_norm.eval()
    with torch.no_grad():
        output = model_norm(x_val)
        loss = loss_fn(output, y_val[:, 1:])
        print(f"Epoch: {epoch}, Loss: {round(loss.item(), 3)}")


model_total = NetworkTotal(x_indices, SIZE=SIZE, DROPOUT=DROPOUT)
optimizer = torch.optim.AdamW(model_total.parameters(), lr=0.0001, weight_decay=0.01)
loss_fn = nn.MSELoss()

model_total.to(device)

# Training loop
epochs = 50
batch_size = 32

for epoch in range(epochs):
    model_total.train()
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i : i + batch_size]
        y_batch = y_train[i : i + batch_size][:, 0][:, np.newaxis]

        optimizer.zero_grad()
        output = model_total(x_batch)
        loss = loss_fn(output, y_batch)
        loss.backward()
        optimizer.step()

    # Validation loss
    model_total.eval()
    with torch.no_grad():
        output = model_total(x_val)
        loss = loss_fn(output, y_val[:, 0][:, np.newaxis])
        print(f"Epoch: {epoch}, Loss: {round(loss.item(), 3)}")


# save the models
torch.save(model_norm.state_dict(), os.path.join(MODEL_FOLDER, "model_norm.pth"))
torch.save(model_total.state_dict(), os.path.join(MODEL_FOLDER, "model_total.pth"))
