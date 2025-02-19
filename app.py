import numpy as np
from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def Home():
    return "Welcome to the Prediction API!"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Extract features
        float_features = [
            float(data["tv"]),
            float(data["radio"])
        ]

        # Reshape features and make a prediction
        features = np.array(float_features).reshape(1, -1)
        prediction = model.predict(features)

        # Return prediction as JSON
        return jsonify({"prediction": float(prediction[0])})

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 400


if __name__ == '__main__':
    app.run(debug=True)
