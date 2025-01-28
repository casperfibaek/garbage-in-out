import os
import numpy as np
import torch
import torch.nn as nn
from model_arch import NetworkNorm, NetworkTotal


torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)

DATA_FOLDER = "./data"
MODEL_FOLDER = "./models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_data():
    DATA_X = np.load(os.path.join(DATA_FOLDER, "x_train.npy"))
    DATA_Y = np.load(os.path.join(DATA_FOLDER, "y_train.npy"))

    # shuffle the data
    indices = np.arange(len(DATA_X))
    np.random.shuffle(indices)
    DATA_X = DATA_X[indices]
    DATA_Y = DATA_Y[indices]

    # normalize the y data
    y_total = (DATA_Y.sum(axis=1) / DATA_X.sum(axis=1))[..., None]
    y_norm = DATA_Y / DATA_Y.sum(axis=1)[:, None]
    DATA_Y = np.concatenate([y_total, y_norm], axis=1)

    # normalize x data
    DATA_X = DATA_X / DATA_X.sum(axis=1)[:, None]

    # split the data into training and validation
    split = int(len(DATA_X) * 0.8)
    x_train, x_val = DATA_X[:split], DATA_X[split:]
    y_train, y_val = DATA_Y[:split], DATA_Y[split:]

    # convert the data to tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    x_train = x_train.to(DEVICE)
    x_val = x_val.to(DEVICE)
    y_train = y_train.to(DEVICE)
    y_val = y_val.to(DEVICE)

    return x_train, x_val, y_train, y_val


def prepare_data_kfold():
    DATA_X = np.load(os.path.join(DATA_FOLDER, "x_train.npy"))
    DATA_Y = np.load(os.path.join(DATA_FOLDER, "y_train.npy"))

    # shuffle the data
    indices = np.arange(len(DATA_X))
    np.random.shuffle(indices)
    DATA_X = DATA_X[indices]
    DATA_Y = DATA_Y[indices]

    # normalize the y data
    y_total = (DATA_Y.sum(axis=1) / DATA_X.sum(axis=1))[..., None]
    y_norm = DATA_Y / DATA_Y.sum(axis=1)[:, None]
    DATA_Y = np.concatenate([y_total, y_norm], axis=1)

    # normalize x data
    DATA_X = DATA_X / DATA_X.sum(axis=1)[:, None]

    # convert the data to tensors
    x_train = torch.tensor(DATA_X, dtype=torch.float32)
    y_train = torch.tensor(DATA_Y, dtype=torch.float32)

    # create five folds
    x_train = torch.split(x_train, len(x_train) // 5)
    y_train = torch.split(y_train, len(y_train) // 5)

    return x_train, y_train


def train_model(
    x_train,
    x_val,
    y_train,
    y_val,
    model_arch,
    size=128,
    dropout=0.1,
    epochs=250,
    bs=32,
    lr=0.0001,
    wd=0.01,
    patience=10,
    sum_loss=False,
):
    model_norm = model_arch(
        x_train.shape[1],
        y_train.shape[1],
        SIZE=size,
        DROPOUT=dropout,
    )
    optimizer = torch.optim.AdamW(model_norm.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()

    model_norm.to(DEVICE)

    best_loss = 100000.0
    counter = 0
    for epoch in range(epochs):
        model_norm.train()
        for i in range(0, len(x_train), bs):
            x_batch = x_train[i : i + bs]

            if sum_loss:
                y_batch = y_train[i : i + bs][:, 0][:, np.newaxis]
            else:
                y_batch = y_train[i : i + bs][:, 1:]

            optimizer.zero_grad()
            output = model_norm(x_batch)
            loss = loss_fn(output, y_batch)
            loss.backward()
            optimizer.step()

        # Validation loss
        model_norm.eval()
        with torch.no_grad():
            output = model_norm(x_val)

            if sum_loss:
                loss = loss_fn(output, y_val[:, 0][:, np.newaxis]).item() * 100.0
            else:
                loss = loss_fn(output, y_val[:, 1:]).item() * 100.0

            if loss < best_loss:
                best_loss = loss
                counter = 0
            else:
                counter += 1

            print(f"Epoch: {epoch + 1}, Loss: {round(loss, 3)}")

            if counter > patience:
                break

    print(
        f"Best Loss: {round(best_loss, 3)} | size: {size} | dropout: {dropout} | lr: {lr} | wd: {wd} | bs: {bs}"
    )

    return model_norm


def train_model_folded(model, sum_loss=False):
    x_train, y_train = prepare_data_kfold()

    models = []
    for i in range(5):
        x_val = x_train[i]
        y_val = y_train[i]

        x_train_fold = torch.cat([x_train[j] for j in range(5) if j != i])
        y_train_fold = torch.cat([y_train[j] for j in range(5) if j != i])

        x_train_fold = x_train_fold.to(DEVICE)
        x_val = x_val.to(DEVICE)
        y_train_fold = y_train_fold.to(DEVICE)
        y_val = y_val.to(DEVICE)

        model_norm = train_model(
            x_train_fold,
            x_val,
            y_train_fold,
            y_val,
            model,
            sum_loss=sum_loss,
        )

        models.append(model_norm)

    return models


if __name__ == "__main__":
    x_train, x_val, y_train, y_val = prepare_data()

    model_norm = train_model(
        x_train,
        x_val,
        y_train,
        y_val,
        NetworkNorm,
        sum_loss=False,
    )

    model_total = train_model(
        x_train,
        x_val,
        y_train,
        y_val,
        NetworkTotal,
        sum_loss=True,
    )

    # save the models
    torch.save(model_norm.state_dict(), os.path.join(MODEL_FOLDER, "model_norm.pth"))
    torch.save(model_total.state_dict(), os.path.join(MODEL_FOLDER, "model_total.pth"))

    model_norm = train_model_folded(NetworkNorm, sum_loss=False)
    model_total = train_model_folded(NetworkTotal, sum_loss=True)

    for i, model in enumerate(model_norm):
        torch.save(
            model.state_dict(), os.path.join(MODEL_FOLDER, f"model_norm_{i}.pth")
        )

    for i, model in enumerate(model_total):
        torch.save(
            model.state_dict(), os.path.join(MODEL_FOLDER, f"model_total_{i}.pth")
        )
