from flask import Flask, request, jsonify, send_from_directory

from inference import predict_output, create_input_tensor

app = Flask(__name__, static_folder="app", static_url_path="")


@app.route("/predict_outputs", methods=["POST"])
def predict_outputs():
    try:
        # Get input data from POST request
        data = request.get_json()

        materials = data["materials"]

        material_ids = []
        material_amounts = []

        for material in materials:
            material_ids.append(int(material["name"]))
            material_amounts.append(int(material["amount"]))

        x_test = create_input_tensor(material_ids, material_amounts)

        # Run prediction using the model
        result = predict_output(x_test)

        # Return prediction results
        return jsonify({"status": "success", "prediction": result})

    except (ValueError, KeyError) as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/")
def serve_static():
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
