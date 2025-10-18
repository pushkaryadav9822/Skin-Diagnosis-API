import os
import io
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import sys

# --- Configuration (Final, simplified settings) ---
# NOTE: Model file MUST be placed in the same folder as this app.py file.
MODEL_PATH = 'mobilenetv2_best.keras'
IMG_SIZE = (224, 224)

# These are the classes in alphabetical order, as determined by Keras:
CLASS_NAMES = ['Chickenpox', 'Measles', 'Monkeypox', 'Normal'] 

app = Flask(__name__)
# Flask will look for templates (like index.html) in the current directory
app.template_folder = os.path.dirname(os.path.abspath(__file__))

model = None

def load_ai_model():
    """Load the pre-trained Keras model once when the Flask app starts."""
    global model
    
    # This flag is required for TF 2.16+ on Apple Silicon
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    
    try:
        print("Attempting to load model...")
        model = load_model(MODEL_PATH)
        print(f"✅ AI Model loaded successfully from: {MODEL_PATH}")
        print("Ready for predictions.")
    except Exception as e:
        print(f"❌ Error loading model. Ensure {MODEL_PATH} is in the same directory. Details: {e}")
        # Exit if the model fails to load, as the API cannot function.
        sys.exit(1)

def model_predict(img_data, model):
    """
    Predicts the class of the image data.
    """
    # 1. Load and preprocess the image
    img = Image.open(io.BytesIO(img_data)).convert('RGB')
    img = img.resize(IMG_SIZE)
    
    # 2. Convert to numpy array and normalize
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) # Add batch dimension
    x = x / 255.0 # Rescale (must match your training)

    # 3. Predict
    preds = model.predict(x)[0]
    
    # 4. Get the result
    predicted_class_index = np.argmax(preds)
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = float(preds[predicted_class_index]) * 100

    return predicted_class_name, confidence

@app.route('/', methods=['GET'])
def index():
    """Serve the basic HTML file to test the API."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to receive an image via POST and return a prediction."""
    
    if model is None:
        return jsonify({'error': 'AI Model not initialized on server.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        try:
            img_bytes = file.read()
            
            # Perform prediction
            result_class, result_confidence = model_predict(img_bytes, model)
            
            # Return the prediction result as JSON
            return jsonify({
                'prediction': result_class,
                'confidence': f"{result_confidence:.2f}%"
            })
        except Exception as e:
            return jsonify({'error': f'Prediction Failed: {str(e)}'}), 500
    
    return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    # Load the model before running the application
    load_ai_model()
    # Run the app. 
    app.run(debug=True, host='0.0.0.0', port=5000)
