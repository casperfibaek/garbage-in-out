import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import pandas as pd

from model_arch import NetworkNorm, NetworkTotal

torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)

MODEL_FOLDER = "./models"
DATA_FOLDER = "./data"
SIZE = 512
DROPOUT = 0.2
X_INDICES = 489
Y_INDICES = 434
THRESHOLD = 0.01

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


def output_id_to_output_name(output_id):
    return unique_outputs[unique_outputs["id"] == output_id]["name"].values[0]


def create_input_tensor(input_ids, input_amounts):
    zero_tensor = torch.zeros(max_input_row_id + 1, dtype=torch.float32)
    for input_id, input_amount in zip(input_ids, input_amounts):
        row_id = input_id_to_row_id(input_id)
        zero_tensor[row_id] = input_amount

    return zero_tensor.unsqueeze(0)


def inference(x_test):
    with torch.no_grad():
        y_norm = model_norm(x_test)
        y_total = model_total(x_test)

    return y_norm, y_total


def predict_output(input_tensor):
    """
    Predict output compositions given an input tensor of materials.

    Args:
        input_tensor (torch.Tensor): Input tensor containing material amounts.

    Returns:
        list: List of dictionaries containing predicted output compositions.
    """
    torch.manual_seed(42)  # Set fixed seed for reproducibility
    y_norm, y_total = inference(input_tensor)
    output_amount = y_total.item() * input_tensor.sum().item()

    # argsort the y_norm tensor
    sorted_result = torch.argsort(y_norm, descending=True)

    results = []
    for i in range(sorted_result.shape[1]):
        result_index = sorted_result[0, i].item()
        result_value = y_norm[0, result_index].item()
        if result_value < THRESHOLD:
            break

        result_id = row_id_to_output_id(result_index)
        result_name = output_id_to_output_name(result_id)
        result_weight = result_value * output_amount

        results.append(
            {
                "id": int(result_id),
                "row_id": result_index,
                "weight": round(result_weight, 2),
                "name": result_name,
            }
        )

    return results


if __name__ == "__main__":
    x_test = create_input_tensor([11, 656], [2650, 2390])
    print(predict_output(x_test))
