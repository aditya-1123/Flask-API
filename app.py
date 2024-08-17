import numpy as np
from flask import Flask, request, jsonify
import pickle
import json

app = Flask(__name__)

model = pickle.load(open('pune_house_price_model.pickle', 'rb'))

with open('columns.json') as f:
    data = json.load(f)
    columns = data.get('data_columns', [])

required_features = set(columns)

@app.route('/')
def home():
    return "Welcome to Pune's House Price Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    try:
        features = np.zeros(len(columns))

        for key, value in data.items():
            if key in required_features:
                index = columns.index(key)
                features[index] = value
            else:
                return jsonify({
                    'error': f"Feature '{key}' not recognized. Available features: {list(required_features)}."
                })

        if features.shape[0] != len(columns):
            return jsonify({
                'error': f"Expected {len(columns)} features, but got {features.shape[0]}."
            })

        prediction = model.predict([features])
        return jsonify({
            'prediction': prediction[0]
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
