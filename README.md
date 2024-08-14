# Image_classification-using-ANN-and-CNN

### Importing Necessary Libraries
```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt
```
- **tensorflow**: A popular machine learning library.
- **keras**: A high-level API within TensorFlow, used to build and train models.
- **datasets**: Provides access to popular datasets.
- **layers**: Used to create neural network layers.
- **models**: Used to create models by stacking layers.
- **numpy**: A library for numerical computations, especially working with arrays.
- **matplotlib.pyplot**: A library for plotting graphs and images.

### Loading the CIFAR-10 Dataset
```python
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
```
- **datasets.cifar10.load_data()**: Loads the CIFAR-10 dataset, which contains 60,000 32x32 color images in 10 classes, with 6,000 images per class. The dataset is split into 50,000 training images and 10,000 test images.

### Checking the Shape of the Data
```python
X_test.shape
X_train.shape
y_train.shape
```
- **X_train.shape**: The shape of the training images (50,000, 32, 32, 3), meaning 50,000 images of 32x32 pixels and 3 color channels (RGB).
- **y_train.shape**: The shape of the training labels (50,000, 1), which indicates the class of each image.

### Reshaping the Labels
```python
y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)
```
- **reshape(-1,)**: Converts the label arrays from a 2D array (e.g., 50,000, 1) to a 1D array (e.g., 50,000,), making them easier to work with.

### Defining Class Names
```python
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
```
- **classes**: A list of class names corresponding to the label indices (0-9).

### Function to Plot a Sample Image
```python
def plot_sample(X, y, index):
    plt.figure(figsize=(15, 2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])
```
- **plot_sample**: This function plots an image from the dataset using `matplotlib`.
- **plt.imshow(X[index])**: Displays the image at the specified index.
- **plt.xlabel(classes[y[index]])**: Labels the image with its corresponding class name.

### Normalizing the Data
```python
X_train = X_train / 255.0
X_test = X_test / 255.0
```
- **Normalization**: The pixel values are divided by 255 to scale them between 0 and 1, which helps in faster and more stable training of the neural network.

### Building a Simple Artificial Neural Network (ANN)
```python
ann = models.Sequential([
    layers.Flatten(input_shape=(32,32,3)),
    layers.Dense(3000, activation='relu'),
    layers.Dense(1000, activation='relu'),
    layers.Dense(10, activation='softmax'),
])
```
- **Sequential**: A linear stack of layers.
- **Flatten**: Converts the 3D image array (32x32x3) into a 1D vector (3072 elements) to feed into the Dense layers.
- **Dense(3000, activation='relu')**: A fully connected layer with 3000 neurons and ReLU activation.
- **Dense(1000, activation='relu')**: A fully connected layer with 1000 neurons and ReLU activation.
- **Dense(10, activation='softmax')**: The output layer with 10 neurons (one for each class) and softmax activation, which converts the outputs into probability distributions.

### Compiling and Training the ANN
```python
ann.compile(optimizer='SGD',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

ann.fit(X_train, y_train, epochs=5)
```
- **optimizer='SGD'**: Stochastic Gradient Descent optimizer.
- **loss='sparse_categorical_crossentropy'**: Loss function used for multi-class classification.
- **metrics=['accuracy']**: Tracks the accuracy during training.
- **ann.fit(X_train, y_train, epochs=5)**: Trains the model on the training data for 5 epochs.

### Evaluating the ANN with Confusion Matrix and Classification Report
```python
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
y_pred = ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print('classification report: \n', classification_report(y_test, y_pred_classes))
```
- **confusion_matrix**: Generates a matrix to evaluate the accuracy of a classification.
- **classification_report**: Provides a detailed classification report with precision, recall, and F1-score.
- **ann.predict(X_test)**: Generates predictions for the test data.
- **np.argmax(element)**: Converts the predicted probabilities into class labels.

### Visualizing the Confusion Matrix
```python
import seaborn as sns
plt.figure(figsize=(14,7))
sns.heatmap(y_pred, annot=True)
plt.ylabel('Truth')
plt.xlabel('Prediction')
plt.title('Confusion matrix')
plt.show()
```
- **sns.heatmap**: Uses Seaborn to create a heatmap visualization of the confusion matrix.
- **plt.show()**: Displays the confusion matrix.

### Building a Convolutional Neural Network (CNN)
```python
cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax'),
])
```
- **Conv2D**: A convolutional layer that applies filters to the input image.
- **MaxPooling2D**: Reduces the spatial dimensions (height and width) by taking the maximum value in a window.
- **Flatten**: Flattens the 2D outputs from the previous layer into a 1D vector.
- **Dense**: Fully connected layers similar to the ANN.

### Compiling and Training the CNN
```python
cnn.compile(optimizer='adam', 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy'])

cnn.fit(X_train, y_train, epochs=10)
```
- **adam**: A more sophisticated optimizer compared to SGD.
- **cnn.fit(X_train, y_train, epochs=10)**: Trains the CNN for 10 epochs.

### Evaluating the CNN
```python
cnn.evaluate(X_test, y_test)
```
- **cnn.evaluate**: Evaluates the CNN on the test data.

### Making Predictions with the CNN
```python
y_pred = cnn.predict(X_test)
y_classes = [np.argmax(element) for element in y_pred]
```
- **y_classes**: Contains the predicted class labels for the test set.

### Visualizing and Understanding Predictions
```python
plot_sample(X_test, y_test, 60)
plot_sample(X_test, y_test, 100)
classes[y_classes[60]]
```
- **plot_sample**: Visualizes a few test images.
- **classes[y_classes[60]]**: Checks the predicted class for a specific image.

### Summary
- **What can be done with this code?**
  - Train both a simple ANN and a more complex CNN on the CIFAR-10 dataset.
  - Evaluate the performance of these models using accuracy, confusion matrices, and classification reports.
  - Visualize sample images and predictions to understand how well the model is performing.
  - The CNN is likely to perform better than the ANN because it is designed to handle image data more effectively through convolutional layers.
