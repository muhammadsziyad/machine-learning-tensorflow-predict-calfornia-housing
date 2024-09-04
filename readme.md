Let's create a TensorFlow project using Keras with the California Housing dataset. This dataset is used for regression tasks, where the goal is to predict the median house value for California districts based on various features.

Project Structure
Here's the project structure for the California Housing dataset:

```css
tensorflow_california_housing/
│
├── data/
│   └── california_housing_data.py  # Script to load California Housing data
├── models/
│   ├── dnn_model.py                # Script to define and compile a DNN model
│   ├── cnn_model.py                # Script to define and compile a CNN model (for experimentation)
├── train.py                        # Script to train the model
├── evaluate.py                     # Script to evaluate the trained model
└── utils/
    └── plot_history.py             # Script to plot training history
```

Step 1: Load California Housing Data
Create a file named california_housing_data.py in the data/ directory to load and preprocess the California Housing dataset.

```python
# data/california_housing_data.py

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data():
    # Load the dataset
    (x_train_full, y_train_full), _ = tf.keras.datasets.california_housing.load_data()

    # Split the dataset into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)

    # Standardize the features (flatten to 2D if necessary)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Ensure the standardized values are <= 2 by scaling down
    scaling_factor = 2 / np.max(np.abs(x_train))
    x_train = x_train * scaling_factor
    x_test = x_test * scaling_factor

    # Reshape back to the original shape if required (keeping 2D as per the StandardScaler requirement)
    # No need to reshape to 3D unless your model specifically requires it.

    return (x_train, y_train), (x_test, y_test)
```

Step 2: Define Models
Below are some models tailored for the California Housing dataset:

1. DNN Model for California Housing

```python
# models/dnn_model.py

import tensorflow as tf
from tensorflow.keras import layers, models

def build_dnn_model(input_shape=(8,)):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)  # Output layer for regression
    ])

    model.compile(optimizer='adam',
                  loss='mse',  # Mean Squared Error for regression
                  metrics=['mae'])  # Mean Absolute Error for evaluation
    return model
```

2. CNN Model for California Housing (Experimental)
Similar to the Boston Housing example, you can experiment with a CNN for tabular data by treating the features as a 1D spatial sequence.

```python
# models/cnn_model.py

import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model(input_shape=(8,)):
    model = models.Sequential([
        layers.Reshape((input_shape[0], 1), input_shape=input_shape),
        layers.Conv1D(32, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)  # Output layer for regression
    ])

    model.compile(optimizer='adam',
                  loss='mse',  # Mean Squared Error for regression
                  metrics=['mae'])  # Mean Absolute Error for evaluation
    return model
```

Step 3: Train the Model
Create a train.py script at the root of the project to load data, build the model, and train it.

```python
# train.py

import tensorflow as tf
#from data.california_housing_data import load_data
#from models.dnn_model import build_dnn_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load California Housing data
(x_train, y_train), (x_test, y_test) = load_data()

# Build the DNN model
model = build_dnn_model()

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

# Train the model
history = model.fit(x_train, y_train, epochs=100, 
                    validation_data=(x_test, y_test),
                    callbacks=[early_stopping, reduce_lr])

# Save the trained model
model.save('dnn_california_housing_model_scaled.keras')

# Save the training history
import pickle
with open('history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
```

Step 4: Evaluate the Model
Create an evaluate.py script to evaluate the trained model on the test data.

```python
# evaluate.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import mse
# from data.california_housing_data import load_data

# Load California Housing data
(x_train, y_train), (x_test, y_test) = load_data()

# Load the trained model
model = keras.models.load_model("dnn_california_housing_model_scaled.keras", custom_objects={'mse': mse})


# Evaluate the model
test_loss, test_mae = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest MAE: {test_mae}')
```

Step 5: Plot Training History
Use the same plot_history.py script to plot the training and validation loss and MAE.

```python
# utils/plot_history.py

import matplotlib.pyplot as plt
import pickle

def plot_history(history_file='history.pkl'):
    with open(history_file, 'rb') as f:
        history = pickle.load(f)

    plt.figure(figsize=(10, 5))
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend(loc='upper right')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(history['mae'], label='MAE')
    plt.plot(history['val_mae'], label='Val MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error (MAE)')
    plt.legend(loc='upper right')
    plt.show()

if __name__ == '__main__':
    plot_history()
```

Step 6: Run the Project
Train the Model: Run the train.py script to start training the model.

```bash
python train.py
```
Evaluate the Model: After training, evaluate the model's performance using evaluate.py.

```bash
python evaluate.py
```

Plot the Training History: Visualize the training history using plot_history.py.

```bash
python utils/plot_history.py
```

