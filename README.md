## README: Brain Tumor Classification Using Deep Learning

### Overview
This project implements a Convolutional Neural Network (CNN) using Keras to classify brain MRI images into four categories:
1. **Glioma Tumor**
2. **Meningioma Tumor**
3. **No Tumor**
4. **Pituitary Tumor**

The dataset used in this project is publicly available and consists of MRI images for training and testing. The primary objective is to automate brain tumor classification, aiding in faster and more accurate diagnosis.

---

### Features
- **Image Preprocessing**: Resizes MRI images to a uniform size of 150x150 pixels and converts them into NumPy arrays.
- **Deep Learning Model**: A CNN architecture with multiple convolutional, pooling, and dropout layers to extract features and prevent overfitting.
- **Training and Validation**: The model is trained on the dataset with 20 epochs, using a validation split of 10%.
- **Evaluation Metrics**:
  - Accuracy
  - Loss
  - Confusion Matrix
  - F1 Score
- **Visualization**:
  - Training and validation accuracy/loss graphs.
  - Confusion matrix for performance evaluation.
  - Predicted vs. true labels for sample test images.

---

### Requirements
#### Libraries
The following Python libraries are required:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `keras`
- `tensorflow`
- `scikit-learn`
- `cv2` (OpenCV)
- `PIL` (Pillow)

Install these libraries using pip if they are not already installed:
```bash
pip install numpy pandas matplotlib seaborn keras tensorflow scikit-learn opencv-python pillow
```

#### Platform
- This code is designed to run on Kaggle's Python environment or any system with Python 3.x installed.

---

### Dataset
The dataset is structured as follows:
```
/brain-tumor-classification-mri/
    ├── Training/
    │   ├── glioma_tumor/
    │   ├── meningioma_tumor/
    │   ├── no_tumor/
    │   └── pituitary_tumor/
    └── Testing/
        ├── glioma_tumor/
        ├── meningioma_tumor/
        ├── no_tumor/
        └── pituitary_tumor/
```

Place the dataset in the directory `/kaggle/input/brain-tumor-classification-mri/`.

---

### How to Run
1. **Clone or Download the Repository**:
   Clone this project or copy the script into your local Python environment.

2. **Set Up the Dataset**:
   Ensure the dataset is placed under `/kaggle/input/brain-tumor-classification-mri/` or modify the script to point to the correct dataset directory.

3. **Run the Script**:
   Execute the script in Kaggle or your local Python environment.

4. **Visualize Results**:
   After training and evaluation, inspect the accuracy/loss plots, confusion matrix, and sample predictions.

---

### Model Architecture
The CNN is structured as follows:
- Input layer: Image size `(150, 150, 3)`
- Convolutional layers: Extract spatial features
- MaxPooling layers: Downsample feature maps
- Dropout layers: Prevent overfitting
- Dense layers: Fully connected layers for classification
- Output layer: Softmax activation for multi-class classification

---

### Results
- **Training and Validation**:
  - The model achieves a certain level of accuracy (can be viewed in the plotted graphs).
  - Validation loss and accuracy provide insights into model performance on unseen data.

- **Confusion Matrix**:
  - Highlights the classification performance for each category.

- **F1 Score**:
  - Weighted F1 Score is calculated to measure the overall effectiveness of the model.

---

### Key Sections of Code
1. **Image Preprocessing**:
   - Loads and resizes images.
   - Converts image labels to one-hot encoding.

2. **Model Training**:
   - Compiles and fits the CNN model using categorical cross-entropy loss and Adam optimizer.

3. **Evaluation and Visualization**:
   - Plots accuracy and loss curves.
   - Displays a confusion matrix and sample test predictions.

---

### Sample Output
- **Accuracy/Loss Graphs**: Visualize model performance over epochs.
- **Confusion Matrix**: Shows true vs. predicted labels for test data.
- **Sample Predictions**: Display sample images with their true and predicted labels.

---

### References
- Dataset: [Kaggle Dataset - Brain Tumor Classification](https://www.kaggle.com/)
- Libraries:
  - [Keras Documentation](https://keras.io/)
  - [TensorFlow Documentation](https://www.tensorflow.org/)
  - [OpenCV Documentation](https://docs.opencv.org/)

---

### Future Work
- Experiment with transfer learning using pre-trained models like VGG16 or ResNet.
- Optimize hyperparameters (e.g., learning rate, batch size).
- Extend the model to handle more categories or larger datasets.

---

### Acknowledgments
This project leverages the power of deep learning to assist in the early detection of brain tumors. Thanks to Kaggle for providing the dataset and Python for its extensive libraries enabling this implementation.
