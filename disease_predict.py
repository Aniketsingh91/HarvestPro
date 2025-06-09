# disease_predict.py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load models
models = {
    'apple': load_model("models/final_apple_model_v2.h5", compile=False),
    'banana': load_model("models/final_banana_model_v2.h5", compile=False),
    'corn': load_model("models/final_corn_model_v2.h5", compile=False),
    'potato': load_model("models/final_potato_model_v2.h5", compile=False),
    'grape': load_model("models/final_grape_model_v2.h5", compile=False)
}

# Compile models
for model in models.values():
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Class labels
labels = {
    'apple': ['Apple Black rot', 'Apple Scab', 'Apple is Healthy', 'Cedar Apple rust'],
    'banana': ['Banan_pestalotiopsis', 'Banana_cordana', 'Banana_healthy', 'Banana_sigatoka'],
    'corn': ['Corn_Cercospora_leaf_spot', 'Corn_Common_rust', 'Corn_Northern_Leaf_Blight', 'Corn_healthy'],
    'potato': ['Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy'],
    'grape': ['Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy']
}


def predict_disease(image_path, crop_type):
    try:
        crop_type = crop_type.lower()

        if crop_type not in models:
            return "Invalid crop type for disease prediction"

        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        model = models[crop_type]
        predictions = model.predict(img_array)[0]
        max_index = np.argmax(predictions)
        label = labels[crop_type][max_index]
        confidence = float(predictions[max_index]) * 100

        return f"{label} ({confidence:.2f}%)"
    except Exception as e:
        return f"Disease Prediction Error: {str(e)}"
