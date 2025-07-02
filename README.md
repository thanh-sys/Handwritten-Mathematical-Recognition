# Handwritten Mathematical Expression Recognition

A system for recognizing and solving handwritten mathematical expressions, including digits (0-9) and operators (+, -, ×, ÷, (, )). Users can draw math expressions via a graphical interface or upload images, and the system detects characters and computes the final result.

---
## 📂 Dataset

We used the dataset from [Handwritten Math Symbols](https://www.kaggle.com/datasets/xainano/handwrittenmathsymbols).

### Dataset Details

- Images are JPG files of size **45x45 pixels**.
- Includes:
  - English alphanumeric symbols.
  - Basic Greek alphabet symbols (e.g. alpha, beta, gamma, mu, sigma, phi, theta).
  - Mathematical operators and set operators.
  - Basic mathematical functions (e.g. log, lim, cos, sin, tan).
  - Advanced math symbols like ∫, ∑, √, δ, etc.
- Excludes Hebrew alphabet characters.

This dataset was originally derived from CROHME datasets, designed for online handwriting recognition tasks. The original CROHME data stored pen trace sequences and timestamps in **inkml** format. In this dataset:
- InkML traces were converted into bitmap images.
- Black pixels represent the handwritten symbol (Region of Interest).
- White pixels are background.
- The images were flattened for ML training, and labels encoded in one-hot format.
![image](https://github.com/user-attachments/assets/5b9e8c4d-e4d7-4346-9434-647fa8e2888f)


## 🎯 Project Objectives

- Build a system that recognizes and evaluates handwritten mathematical expressions.
- Process images containing mathematical symbols to clean the dataset and improve model performance.
- Train a Convolutional Neural Network (CNN) to recognize handwritten math characters.
- Provide a user-friendly GUI for drawing or uploading expressions.
- Parse and evaluate recognized expressions to output the result.

---

## 📝 Main Features

✅ **Data Processing**  
- Remove duplicate images using hashing.
- Detect corrupted or outlier images.
- Preprocess images (resizing, binarization).

![image](https://github.com/user-attachments/assets/bc2ba486-3a66-4c1d-b389-22768c9c5bb2)


✅ **Model Training**  
- Use CNNs built with TensorFlow/Keras.
- Classify handwritten digits and operators.
- Evaluate using classification reports and confusion matrices.

✅ **Image Processing Techniques**  
- Adaptive thresholding.
- Morphological operations: dilation, closing, thinning.
- Contour detection and segmentation.

✅ **User Interface (GUI)**  
- Built with Tkinter.
- Allow users to draw mathematical expressions.
- Allow uploading images containing expressions.
- Display recognized expression and computed result.

✅ **Expression Evaluation**  
- Detect characters from the image.
- Reconstruct the mathematical expression string (e.g. `"2+3"`).
- Compute the result using Python’s `eval`.

---

## 🛠️ Technologies and Libraries Used

### Programming Language
- Python

### Image Processing
- OpenCV (`cv2`) — image thresholding, morphology, contour detection.
- Pillow (PIL) — drawing and displaying images in the GUI.
- scikit-image — HOG features for outlier detection.

### Machine Learning
- TensorFlow / Keras — build and train CNNs.
- NumPy — numerical operations and data normalization.
- scikit-learn — train-test split, class weighting, evaluation metrics.

### Visualization
- Matplotlib — plotting label distributions, displaying images, confusion matrices.
- Seaborn — advanced plotting, confusion matrices.

### GUI Development
- Tkinter — build graphical interface for drawing/uploading images.

### Utilities
- hashlib — detect duplicate images.
- collections — count images per label.
- pandas — assist with data manipulation (minor usage).
- shutil — archive/unarchive datasets.
- os — file and folder handling.

---

## 📊 Results

- Successfully recognized digits and operators in isolated character images.
- Built an end-to-end pipeline: draw → recognize → compute result.
- Achieved high classification accuracy on test data.

  
![image](https://github.com/user-attachments/assets/5431e656-dfa6-4e7c-80f6-499351868a7f)

---
## 🚀 Run GUI Demo (Local)

### 1. Install Required Libraries

Make sure you have Python (≥ 3.7) installed.  
Install the necessary libraries:

```bash
pip install opencv-contrib-python tensorflow keras pillow numpy matplotlib seaborn scikit-learn
```
Ensure the following trained model files are present in your project directory:

model_final.json

model_final.weights.h5

These include the architecture and weights of the trained CNN model.
A Tkinter window will open.

You can:

Draw a handwritten mathematical expression on the canvas.

Upload an image containing a handwritten expression.

Click the “Extract” button to recognize the expression and compute the result.

The recognized expression and its evaluated result will be displayed on the screen.


DEMO: (Model works best with small image and draw image small) 
![image](https://github.com/user-attachments/assets/677fca54-36a3-46ae-a0f5-68decb73f682)

![image](https://github.com/user-attachments/assets/6b278faf-8bb4-439b-bfe9-a9adfe4e7afc)

