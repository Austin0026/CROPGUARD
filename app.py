import os
import tensorflow as tf
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit file size to 16 MB

# Ensure the 'uploads' directory exists
os.makedirs('uploads', exist_ok=True)

# Load the model
try:
    model = tf.keras.models.load_model('model.h5')
    print('Model loaded. Check http://127.0.0.1:5000/')
except Exception as e:
    print(f"Error loading model: {e}")

# Labels for the model predictions
labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

def getResult(image_path):
    try:
        # Load and preprocess the image
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(225, 225))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = x.astype('float32') / 255.0
        x = np.expand_dims(x, axis=0)

        # Make predictions
        predictions = model.predict(x)[0]
        print("Raw predictions:", predictions)

        # Get the predicted label
        predicted_label = np.argmax(predictions)
        result = labels.get(predicted_label, "Invalid")
        
        # Debugging output to confirm label matching
        print(f"Predicted index: {predicted_label}, Result: {result}")
        
        return result
    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Invalid"

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        # Save the uploaded file
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Get prediction result
        result = getResult(file_path)
        
        # Return the prediction result directly
        return f"Predicted Label: {result}"
    return None

if __name__ == '__main__':
    app.run(debug=True)
