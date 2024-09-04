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