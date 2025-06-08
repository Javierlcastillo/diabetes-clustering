from flask import Flask, request, jsonify
import numpy as np
import pickle
from kmodes.kprototypes import KPrototypes
import os

# Cargar el modelo
with open("kproto_model.pkl", "rb") as f:
    kproto = pickle.load(f)

app = Flask(__name__)

# Ruta para verificar que el servicio está funcionando
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

# Ruta para predecir el cluster
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        # Entrada esperada: ['HighBP', 'HighChol', 'GenHlth', 'Sex', 'Diabetes', 'Age', 'BMI']
        input_data = np.array([[
            int(data['HighBP']),
            int(data['HighChol']),
            int(data['GenHlth']),
            int(data['Sex']),
            int(data['Diabetes']),
            int(data['Age']),
            float(data['BMI'])
        ]], dtype=object)

        # Índices categóricos
        categorical_cols = [0, 1, 2, 3, 4, 5]

        cluster = kproto.predict(input_data, categorical=categorical_cols)[0]
        return jsonify({'cluster': int(cluster)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
