from flask import Flask, request, jsonify, render_template
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

crop_details = {
    'Rice': {
        'model' : 'models/riceModel.h5',
        'class_names' : ['Bacterialblight','Blast','Brownspot','Healthy','Tungro']
    },
    'Banana': {
        'model' : 'models/bananaModel.h5',
        'class_names' : ['Black Sigatoka Disease','Bract Mosaic Virus Disease','Healthy leaf','Insect Pest Disease','Moko Disease','Panama Disease','Yellow Sigatoka Disease']
    },
    'Sugarcane': {
        'model' : 'models/sugarcaneModel.h5',
        'class_names' : ['Healthy','Mosaic','RedRot','Rust','Yellow']
    },
    'Groundnut': {
        'model' : 'models/groundnutModel.h5',
        'class_names' : ['Early leaf spot','Early rust','Healthy leaf','Late leaf spot','Nutrition deficiency','Rust']
    }
}

# Function to preprocess the image
def preprocess_image(img):
    img = image.load_img(img, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.
    return img

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():

    crop_selected = crop_details[request.form['crop_type']]

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
        
    if file:
        img_path = 'static/crop_image.jpg'  # Temporarily save image
        file.save(img_path)
        img = preprocess_image(img_path)
        model = load_model(crop_selected['model'])
        model.make_predict_function()  # Necessary for multi-threaded execution
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)
        return jsonify({'plant status': crop_selected['class_names'][predicted_class]})

if __name__ == '__main__':
    app.run(debug=False)
