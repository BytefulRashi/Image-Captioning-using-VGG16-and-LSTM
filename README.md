# Image Caption Generator

This project provides a web application that generates captions for images using a pre-trained deep learning model. The application allows users to upload an image and receive a descriptive caption as output.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Directory Structure](#directory-structure)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction

This application uses a combination of Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) to generate captions for images. The CNN extracts features from images, while the RNN generates descriptions based on these features.

## Features

- Upload an image and get an automatically generated caption.
- Clean and modern UI using HTML, CSS, and JavaScript.
- Uses VGG16 for image feature extraction and an LSTM-based model for caption generation.

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/image-caption-generator.git
    cd image-caption-generator
    ```

2. **Set up a virtual environment and activate it:**
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

4. **Place the pre-trained models and tokenizer in the project directory:**
    - `best_model.h5`
    - `tokenizer.pkl`
    - `features.pkl`

## Usage

1. **Run the Flask application:**
    ```sh
    python app.py
    ```

2. **Open your browser and navigate to:**
    ```
    http://127.0.0.1:5000/
    ```

3. **Upload an image to generate a caption.**

## Model Training

The model training involves several steps including feature extraction, caption processing, and model training. Here's a brief overview:

1. **Feature Extraction:**
   Using VGG16 to extract features from images.

2. **Caption Processing:**
   Cleaning and tokenizing the captions, and creating sequences.

3. **Model Training:**
   Training a model with an LSTM network to generate captions.

For detailed steps and code, please refer to the `6_Image_Caption_Generation.ipynb` notebook.

## Directory Structure

```
image-caption-generator/
│
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── templates/
│   └── upload.html           # HTML template for the upload form
├── static/
│   ├── uploads/              # Directory to store uploaded images
│   └── styles.css            # CSS for styling the application
├── models/                   # Directory containing model training code
│   └── Image_Caption_Generation.ipynb  # Jupyter notebook for model training
├── best_model.h5             # Pre-trained caption generation model
├── tokenizer.pkl             # Tokenizer used for text processing
└── features.pkl              # Pre-computed image features

```


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- The dataset used for training the model is the Flickr8k dataset.
- The VGG16 model used for feature extraction is from Keras applications.
- Special thanks to the contributors of various tutorials and repositories that inspired this project.
