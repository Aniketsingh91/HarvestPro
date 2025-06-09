# --- app.py ---
from flask import Flask, render_template, request
from crop_predict import predict_crop
from fertilizer_predict import predict_fertilizer
from disease_predict import predict_disease
import os
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template("pro_v5.html")

@app.route('/crop_predict', methods=['POST'])
def predict_crop_route():
    result = predict_crop(request.form)  # List of (crop, %)
    return render_template("pro_v5.html", crop_result=result)

@app.route('/predict_fertilizer', methods=['POST'])
def fertilizer():
    form_data = {
        key: value.lower() if key in ['soil_type', 'crop_type'] else value
        for key, value in request.form.items()
    }
    result = predict_fertilizer(form_data)  # Now returns list of (fertilizer, %)
    return render_template("pro_v5.html", fert_result=result)

@app.route('/predict_disease', methods=['POST'])
def disease():
    image = request.files.get('image_file')
    crop_type = request.form.get('crop_type')

    if not image or not crop_type:
        return "Missing crop type or image file", 400

    filename = f"{uuid.uuid4()}_{secure_filename(image.filename)}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)

    result = predict_disease(filepath, crop_type.lower())
    return render_template("pro_v5.html", disease_result=result)

if __name__ == '__main__':
    app.run(debug=True)
