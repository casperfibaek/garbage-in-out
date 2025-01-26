import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import pandas as pd

from model_arch import NetworkNorm, NetworkTotal

MODEL_FOLDER = "./models"
DATA_FOLDER = "./data"
SIZE = 512
DROPOUT = 0.2
X_INDICES = 489
Y_INDICES = 434

model_norm = NetworkNorm(X_INDICES, Y_INDICES, SIZE=SIZE, DROPOUT=DROPOUT)
model_total = NetworkTotal(X_INDICES, SIZE=SIZE, DROPOUT=DROPOUT)

# Load the model
model_norm.load_state_dict(
    torch.load(os.path.join(MODEL_FOLDER, "model_norm.pth"), weights_only=True)
)
model_total.load_state_dict(
    torch.load(os.path.join(MODEL_FOLDER, "model_total.pth"), weights_only=True)
)

# Set the model to evaluation mode
model_norm.eval()
model_total.eval()

# load unique inputs and outputs
unique_inputs = pd.read_csv(os.path.join(DATA_FOLDER, "unique_inputs.csv"), sep=";")
unique_outputs = pd.read_csv(os.path.join(DATA_FOLDER, "unique_outputs.csv"), sep=";")

max_input_row_id = unique_inputs["row_id"].max()
max_output_row_id = unique_outputs["row_id"].max()


def row_id_to_input_id(row_id):
    return unique_inputs[unique_inputs["row_id"] == row_id]["id"].values[0]


def row_id_to_output_id(row_id):
    return unique_outputs[unique_outputs["row_id"] == row_id]["id"].values[0]


def input_id_to_row_id(input_id):
    return unique_inputs[unique_inputs["id"] == input_id]["row_id"].values[0]


def output_id_to_row_id(output_id):
    return unique_outputs[unique_outputs["id"] == output_id]["row_id"].values[0]


def create_input_tensor(input_ids, input_amounts):
    zero_tensor = torch.zeros(max_input_row_id + 1, dtype=torch.float32)
    for input_id, input_amount in zip(input_ids, input_amounts):
        row_id = input_id_to_row_id(input_id)
        zero_tensor[row_id] = input_amount

    return zero_tensor.unsqueeze(0)


x_test = create_input_tensor([145], [5860])


def inference(x_test):
    with torch.no_grad():
        y_norm = model_norm(x_test)
        y_total = model_total(x_test)

    return y_norm, y_total


y_norm, y_total = inference(x_test)
output_amount = y_total.item() * x_test.sum().item()

# top three input ids
top_three_row_ids = torch.topk(y_norm, 3).indices.squeeze().tolist()
top_three_values = torch.topk(F.softmax(y_norm), 3).values.squeeze().tolist()
for row_id in top_three_row_ids:
    input_id = row_id_to_input_id(row_id)
    value = round(top_three_values.pop(0), 3)
    print(f"Input ID: {input_id}, Row ID: {row_id}, Value: {value}")

pdb.set_trace()
