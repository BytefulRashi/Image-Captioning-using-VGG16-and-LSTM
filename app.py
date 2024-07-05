from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import seaborn as sns

application = Flask(__name__)
app = application
app.config['UPLOAD_FOLDER'] = 'uploads'


# Load the fixed tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)


# Load the trained model
model = load_model('best_model.h5')

# Load the VGG16 model
vgg_model = VGG16()
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Function to generate caption
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and caption generation
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Load and preprocess the image
        image = load_img(file_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = vgg_model.predict(image, verbose=0)
        
        # Generate caption
        max_length = 34  # Set your max_length here
        predicted_caption = predict_caption(model, feature, tokenizer, max_length)
        predicted_caption = ' '.join(predicted_caption.split()[1:-1])  # Remove startseq and endseq
        
        # Plotting example using Seaborn (not directly related to captioning)
        # Replace with your specific plotting logic
        sns.set_theme(style="whitegrid")
        tips = sns.load_dataset("tips")
        sns.barplot(x="day", y="total_bill", data=tips)
        plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'plot.png'))
        plot_path = url_for('static', filename='plot.png')
        
        return render_template('result.html', caption=predicted_caption, image_url=file_path, plot_url=plot_path)

if __name__ == '__main__':
    app.run(debug=True)
