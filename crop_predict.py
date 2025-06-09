import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open('models/Crop_model_scaled_v5.pkl', 'rb'))
scaler = pickle.load(open('models/crop_scaler_v5.pkl', 'rb'))

def predict_crop(form_data):
    """
    Predict the top 3 best crops with confidence scores.
    """
    try:
        # Extract and scale features
        features = [
            float(form_data['nitrogen']),
            float(form_data['phosphorus']),
            float(form_data['potassium']),
            float(form_data['temperature']),
            float(form_data['humidity']),
            float(form_data['ph']),
            float(form_data['rainfall'])
        ]
        scaled_features = scaler.transform([features])

        # If model supports predict_proba, get top 3 crops
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(scaled_features)[0]
            class_labels = model.classes_
            top_indices = np.argsort(proba)[::-1][:4]
            top_crops = [(class_labels[i], round(proba[i] * 100, 2)) for i in top_indices]
            return top_crops
        else:
            prediction = model.predict(scaled_features)
            return [(prediction[0], 100.0)]

    except Exception as e:
        return [("Prediction Error", str(e))]
