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