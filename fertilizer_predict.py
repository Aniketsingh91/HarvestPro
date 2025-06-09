from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np

# Load the trained fertilizer classification model
model = joblib.load("models/classifier.pkl")

# Fertilizer mapping
ferti_map = {
    0: '10-26-26', 1: '10-26-26', 2: '14-14-14', 3: '14-35-14',
    4: '15-15-15', 5: '17-17-17', 6: '20-20', 7: '28-28',
    8: 'DAP', 9: 'Potassium chloride', 10: 'Potassium sulfate',
    11: 'Superphosphate', 12: 'TSP', 13: 'Urea'
}

# Encoding dictionaries
soil_map = {'sandy': 4, 'loamy': 2, 'black': 0, 'red': 3, 'clay': 1, 'clayey': 1}
crop_type_map = {
    'barley': 0, 'cotton': 1, 'ground nuts': 2, 'maize': 3, 'millets': 4,
    'oilseeds': 5, 'paddy': 6, 'pulses': 7, 'sugarcane': 8, 'tobacco': 9,
    'wheat': 10, 'coffee': 11, 'kidneybeans': 12, 'orange': 13,
    'pomegranate': 14, 'rice': 15, 'watermelon': 16
}

def predict_fertilizer(data):
    try:
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        moisture = float(data['moisture'])
        nitrogen = float(data['nitrogen'])
        potassium = float(data['potassium'])
        phosphorous = float(data['phosphorous'])

        soil_encoded = soil_map.get(data['soil_type'].lower())
        crop_type_encoded = crop_type_map.get(data['crop_type'].lower())

        if soil_encoded is None or crop_type_encoded is None:
            return "Error: Invalid soil type or crop type."

        features = np.array([[temperature, humidity, moisture, soil_encoded, crop_type_encoded,
                              nitrogen, potassium, phosphorous]])

        # Predict probabilities
        probabilities = model.predict_proba(features)[0]

        # Get indices of top 3 predictions
        top_indices = np.argsort(probabilities)[::-1][:4]

        top_ferts = [
            (ferti_map.get(idx, "Unknown"), round(probabilities[idx] * 100, 2))
            for idx in top_indices
        ]

        return top_ferts

    except Exception as e:
        return f"Prediction Error: {str(e)}"
