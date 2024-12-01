from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import numpy as np
import cv2
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

app = Flask(__name__)

MODEL_PATH = 'knee_final.h5'

model = load_model(MODEL_PATH)
print('Model loaded. Start serving...')

def preprocess_image(image_path):
    img_size = (224, 224)
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.resize(img, img_size)
        img = preprocess_input(img) 
        img = np.expand_dims(img, axis=0)
        return img
    else:
        return None
    
def predict_image(image_path):
    preprocessed_image = preprocess_image(image_path)
    if preprocessed_image is not None:
        predictions = model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions[0])
        reverse_mapping = {0: 0, 1: 2, 2: 4}
        predicted_label = reverse_mapping[predicted_class]
        return predicted_label
    else:
        return "Error: Could not load image."

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    predicted_label = predict_image(image_path)
    print(predicted_label)
    if predicted_label == 0:
        label = "Normal"
    elif predicted_label == 2:
        label = "Knee Pain"
    else:
        label = "Severe Knee Pain"
    # return render_template('index.html', prediction=label)
    return jsonify({"message": "File berhasil diterima", "label": label}), 200

if __name__ == '__main__':
    app.run(port=3000, debug=True)