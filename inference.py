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
X_INDICES = 489
Y_INDICES = 434
THRESHOLD = 0.01
FOLDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_norm = []
models_total = []

# Create the models
for i in range(FOLDS):
    model_norm = NetworkNorm(X_INDICES, Y_INDICES)
    model_total = NetworkTotal(X_INDICES, Y_INDICES)

    # Load the model
    model_norm.load_state_dict(
        torch.load(
            os.path.join(MODEL_FOLDER, f"model_norm_{i}.pth"),
            weights_only=True,
        )
    )
    model_total.load_state_dict(
        torch.load(
            os.path.join(MODEL_FOLDER, f"model_total_{i}.pth"),
            weights_only=True,
        )
    )

    # Set the model to evaluation mode
    model_norm.eval()
    model_total.eval()

    # transfer the model to the GPU
    model_norm.to(DEVICE)
    model_total.to(DEVICE)

    models_norm.append(model_norm)
    models_total.append(model_total)


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

    # Normalize the input tensor
    zero_tensor = zero_tensor / zero_tensor.sum()

    return zero_tensor.unsqueeze(0)


def inference(x_test):
    x_test = x_test.to(DEVICE)

    with torch.no_grad():
        y_norm_results = []
        y_total_results = []

        for model_norm, model_total in zip(models_norm, models_total):
            y_norm = model_norm(x_test)
            y_total = model_total(x_test)

            y_norm_results.append(y_norm)
            y_total_results.append(y_total)

    # TODO: median or mean?
    # y_norm = torch.stack(y_norm_results).mean(dim=0)
    # y_total = torch.stack(y_total_results).mean(dim=0)

    y_norm = torch.stack(y_norm_results).median(dim=0)[0]
    y_total = torch.stack(y_total_results).median(dim=0)[0]

    # calculate the standard deviation
    y_norm_std = torch.stack(y_norm_results).std(dim=0)
    y_total_std = torch.stack(y_total_results).std(dim=0)

    return y_norm, y_total, y_norm_std, y_total_std


def predict_output(input_tensor, material_amounts):
    """
    Predict output compositions given an input tensor of materials.

    Args:
        input_tensor (torch.Tensor): Input tensor containing material amounts.

    Returns:
        list: List of dictionaries containing predicted output compositions.
    """
    torch.manual_seed(42)  # Set fixed seed for reproducibility
    y_norm, y_total, y_norm_std, y_total_std = inference(input_tensor)
    output_amount = y_total.item() * torch.tensor(material_amounts).sum()

    # argsort the y_norm tensor
    sorted_result = torch.argsort(y_norm, descending=True)

    results = []
    for i in range(sorted_result.shape[1]):
        result_index = sorted_result[0, i].item()
        result_value = y_norm[0, result_index].item()
        result_value_std = y_norm_std[0, result_index].item()

        if result_value < THRESHOLD:
            break

        result_id = row_id_to_output_id(result_index)
        result_name = output_id_to_output_name(result_id)
        result_weight = result_value * output_amount.item()

        results.append(
            {
                "id": int(result_id),
                "row_id": result_index,
                "weight": round(result_weight, 1),
                "std": round(result_weight * result_value_std, 1),
                "name": result_name,
            }
        )

    # pdb.set_trace()
    return results


if __name__ == "__main__":
    materials = [11, 656]
    amounts = [2650, 2390]
    x_test = create_input_tensor(materials, amounts)
    print(predict_output(x_test, amounts))
