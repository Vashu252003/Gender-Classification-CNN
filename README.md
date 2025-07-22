# Gender Classification using Convolutional Neural Networks (CNN)

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-3.x-red?style=for-the-badge&logo=keras)

This project implements a deep learning model to classify a person's gender (male or female) from facial images. The model is a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

---

## 📋 Table of Contents

- [Key Features](#-key-features)
- [Results and Demonstration](#-results-and-demonstration)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Setup and Installation](#-setup-and-installation)
- [Usage](#-usage)
- [Contributing](#-contributing)
- [Contact](#-contact)

---

## ✨ Key Features

- **Deep CNN Model**: Utilizes a robust CNN architecture for high-accuracy feature extraction.
- **Data Augmentation**: Employs Keras's `ImageDataGenerator` to create more training data from existing images, preventing overfitting and improving model generalization.
- **Training and Validation Plots**: Generates and saves plots for accuracy and loss to visualize the model's performance over time.
- **Modular and Readable Code**: The project is structured for clarity and ease of understanding.

---

## 📊 Results and Demonstration

### Model Performance

The model achieves high accuracy on both the training and validation sets after 50 epochs.
The plot below shows the learning curves.

![alt text](imagess/accuracy_plot.png)

### Example Prediction

Here is an example of the model classifying a new, unseen image from the test set.

![alt text](imagess/prediction_example.png)

---

## 🧠 Model Architecture

The model is a `Sequential` stack of layers, primarily composed of:

- **5 Convolutional Layers (`Conv2D`)**: Responsible for learning hierarchical features from the images.
- **Batch Normalization**: Used after convolutional layers to stabilize and accelerate training.
- **Max Pooling (`MaxPooling2D`)**: Reduces the spatial dimensions of the feature maps.
- **3 Fully Connected Layers (`Dense`)**: Perform the final classification based on the learned features.
- **Dropout**: Included in the dense layers to reduce overfitting.
- **Sigmoid Activation**: The final output layer uses a sigmoid function to produce a probability score for binary classification.

---

## 📂 Dataset

This model was trained using a large dataset of facial images. Due to its size, the dataset is **not included** in this repository.

### Dataset Setup

1.  **Download the Dataset**: You can find the dataset on Kaggle: [Gender Classification Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset).
2.  **Organize the Data**: After downloading, you must structure the images inside the `data/` folder as follows for the script to work correctly:

    ```
    project-root/
    └── data/
        ├── Train/
        │   ├── Female/
        │   │   ├── 000001.jpg
        │   │   └── ...
        │   └── Male/
        │       ├── 000002.jpg
        │       └── ...
        ├── Validation/
        │   ├── Female/
        │   └── Male/
        └── Test/
            ├── Female/
            └── Male/
    ```

---

## 🛠️ Setup and Installation

Follow these steps to set up the project on your local machine.

### Prerequisites

- Python 3.9+
- Pip (Python package installer)

### Installation Steps

1.  **Clone the Repository**

    ```sh
    git clone [https://github.com/Vashu252003/Gender-Classification-CNN.git](https://github.com/Vashu252003/Gender-Classification-CNN.git)
    cd Gender-Classification-CNN
    ```

2.  **Create and Activate a Virtual Environment**

    ```sh
    # Create the virtual environment
    python -m venv .venv

    # Activate it (Windows)
    .\.venv\Scripts\activate

    # Activate it (macOS/Linux)
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    Install all the required packages from the `req.txt` file.
    ```sh
    pip install -r req.txt
    ```

---

## 🚀 Usage

### Training the Model

To train the model from scratch, simply run the `main.py` script.

```sh
python main.py
```
