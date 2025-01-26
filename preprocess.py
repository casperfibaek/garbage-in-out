# This file preprocesses the input data for the model
# It reads the input data from the excel file and preprocesses it

import os
import pandas as pd
import numpy as np
import pdb

from tqdm import tqdm


DATA_FOLDER = "data/"
DATA_FILE = "production.xlsx"


def preprocess_data_01():
    # Read the input data
    input_data_raw = pd.read_excel(os.path.join(DATA_FOLDER, DATA_FILE))

    # Create a temporary dataframe to store the preprocessed data
    temp_cols = [
        "transaction_id",
        "input_id",
        "input_id_row",
        "output_id",
        "output_id_row",
        "input_amount",
        "output_amount",
    ]
    temp_dtypes = {
        "transaction_id": np.int64,
        "input_id": np.int64,
        "input_id_row": np.int64,
        "output_id": np.int64,
        "output_id_row": np.int64,
        "input_amount": np.float64,
        "output_amount": np.float64,
    }
    temp_df = pd.DataFrame(columns=temp_cols)
    temp_df = temp_df.astype(temp_dtypes)

    # Filter the rows where Transaktions_ID does not start with LJ or PROD
    input_data_lj = input_data_raw[
        input_data_raw["Transaktions_ID"].str.startswith("LJ")
    ]
    input_data_prod = input_data_raw[
        input_data_raw["Transaktions_ID"].str.startswith("PROD")
    ]
    input_data = pd.concat([input_data_lj, input_data_prod])

    unique_inputs_id = []
    unique_outputs_id = []
    unique_inputs = []
    unique_outputs = []

    # Iterate over the rows of the input data
    for _index, row in tqdm(
        input_data.iterrows(), total=input_data.shape[0], desc="Preprocessing data"
    ):
        # Get the transaction id
        transaction_id = row["Transaktions_ID"]

        # strip non numeric characters from the transaction id
        if transaction_id.startswith("PROD"):
            transaction_id = int(
                "".join(filter(str.isdigit, transaction_id.split(" - ")[0]))
            )
        else:
            transaction_id = int("".join(filter(str.isdigit, transaction_id)))

        # Get the input and output ids
        input_str = row["Input varer"]
        output_str = row["Output varer"]

        # If row starts with "S" is manpower and not useful
        if input_str.startswith("S") or output_str.startswith("S"):
            continue

        # Get the input and output amounts
        input_amount = int(row["Input kg"])
        output_amount = int(row["Output kg"])

        # Rows that have zero input or output amounts are not useful
        if input_amount == 0 or output_amount == 0:
            continue

        # Rows that have higher output than input are not useful
        if output_amount > input_amount:
            continue

        # Get the input and output names
        input_name = input_str[10:]
        output_name = output_str[10:]

        # Get the input and output ids
        input_id = int(input_str[1:7])
        output_id = int(output_str[1:7])

        # If unique, add to the list
        if input_id not in unique_inputs_id:
            unique_inputs.append({"name": input_name, "id": input_id})
            unique_inputs_id.append(input_id)
        if output_id not in unique_outputs_id:
            unique_outputs.append({"name": output_name, "id": output_id})
            unique_outputs_id.append(output_id)

        # Create a row for the temp_df
        row_data = {
            "transaction_id": transaction_id,
            "input_id": input_id,
            "input_id_row": -1,
            "output_id": output_id,
            "output_id_row": -1,
            "input_amount": input_amount,
            "output_amount": output_amount,
        }
        temp_row = pd.DataFrame([row_data], columns=temp_cols)
        temp_row = temp_row.astype(temp_dtypes)

        # Add row to the df
        temp_df = pd.concat([temp_df, temp_row])

    # Convert the temp_df to the correct data types
    temp_df.astype(temp_dtypes)

    # sort the unique inputs and outputs by id
    unique_inputs = sorted(unique_inputs, key=lambda x: x["id"])
    unique_outputs = sorted(unique_outputs, key=lambda x: x["id"])

    # add a row id to the unique inputs and outputs
    for index, input_data in enumerate(unique_inputs):
        input_data["row_id"] = index

    for index, output_data in enumerate(unique_outputs):
        output_data["row_id"] = index

    # Save unique inputs and outputs to a file for future use
    unique_inputs_df = pd.DataFrame(unique_inputs)
    unique_outputs_df = pd.DataFrame(unique_outputs)
    unique_inputs_df.astype({"id": np.int64, "row_id": np.int64, "name": str})
    unique_outputs_df.astype({"id": np.int64, "row_id": np.int64, "name": str})

    unique_inputs_df.to_csv(
        os.path.join(DATA_FOLDER, "unique_inputs.csv"),
        index=False,
        sep=";",
    )
    unique_outputs_df.to_csv(
        os.path.join(DATA_FOLDER, "unique_outputs.csv"),
        index=False,
        sep=";",
    )

    temp_df.reset_index(drop=True, inplace=True)
    # loop over the temp_df and replace the input and output ids with the row ids
    for index, row in tqdm(
        temp_df.iterrows(), total=temp_df.shape[0], desc="Replacing ids"
    ):
        input_id = row["input_id"]
        output_id = row["output_id"]

        # find unique input and output row ids
        input_row_id = unique_inputs_df[unique_inputs_df["id"] == input_id][
            "row_id"
        ].values[0]
        output_row_id = unique_outputs_df[unique_outputs_df["id"] == output_id][
            "row_id"
        ].values[0]

        # update the row
        temp_df.at[index, "input_id_row"] = input_row_id
        temp_df.at[index, "output_id_row"] = output_row_id

    # Save the preprocessed data to a file
    temp_df.to_csv(
        os.path.join(DATA_FOLDER, "preprocessed_data_01.csv"),
        index=False,
        sep=";",
    )


def preprocess_data_02():
    # Read the input data
    data = pd.read_csv(
        os.path.join(DATA_FOLDER, "preprocessed_data_01.csv"),
        sep=";",
    )
    unique_inputs = pd.read_csv(
        os.path.join(DATA_FOLDER, "unique_inputs.csv"),
        sep=";",
    )
    unique_outputs = pd.read_csv(
        os.path.join(DATA_FOLDER, "unique_outputs.csv"),
        sep=";",
    )

    dtypes = {
        "transaction_id": np.int64,
        "input_id": np.int64,
        "input_id_row": np.int64,
        "output_id": np.int64,
        "output_id_row": np.int64,
        "input_amount": np.float64,
        "output_amount": np.float64,
    }

    # Get the max values of the row ids
    input_max_val = unique_inputs["row_id"].max()
    output_max_val = unique_outputs["row_id"].max()

    # Get all unique transaction ids
    unique_transaction_ids = data["transaction_id"].unique()

    x_train = []
    y_train = []

    # loop the transactions
    for transaction_id in tqdm(
        unique_transaction_ids, total=len(unique_transaction_ids), desc="Processing"
    ):
        # get the rows for the transaction
        transaction_data = data[data["transaction_id"] == transaction_id]
        transaction_data.reset_index(drop=True, inplace=True)
        transaction_data = transaction_data.astype(dtypes)

        # create a destination array for the transaction
        transaction_input = np.zeros((input_max_val + 1,), dtype=np.float32)
        transaction_output = np.zeros((output_max_val + 1,), dtype=np.float32)

        input_processed = []
        output_processed = []

        total_input_amount = 0
        total_output_amount = 0

        # loop the rows of the transaction
        for _index, row in transaction_data.iterrows():
            input_id_row = int(row["input_id_row"])
            output_id_row = int(row["output_id_row"])
            input_amount = row["input_amount"]
            output_amount = row["output_amount"]

            total_input_amount += input_amount
            total_output_amount += output_amount

            # check if the input and output ids are already processed
            if input_id_row not in input_processed:
                transaction_input[input_id_row] = input_amount

            if output_id_row not in output_processed:
                transaction_output[output_id_row] = output_amount

        # check if the transaction is valid
        if (
            total_output_amount > total_input_amount
            or transaction_output.sum() > transaction_input.sum()
        ):
            continue

        x_train.append(transaction_input)
        y_train.append(transaction_output)

    # convert the data to numpy arrays
    x_train = np.array(x_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)

    # save the data to files
    np.save(os.path.join(DATA_FOLDER, "x_train.npy"), x_train)
    np.save(os.path.join(DATA_FOLDER, "y_train.npy"), y_train)


if __name__ == "__main__":
    preprocess_data_01()
    preprocess_data_02()
