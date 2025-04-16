# Import necessary libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error

# Load the California Housing dataset
california = fetch_california_housing()

# Convert the data into a DataFrame for easy manipulation
X = pd.DataFrame(california.data, columns=california.feature_names)
y = pd.DataFrame(california.target, columns=['Price'])

# Preprocess the data (Scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the DNN Model
model = Sequential()

# Add input layer with 8 features (the number of features in the dataset)
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))

# Add hidden layers
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

# Output layer with 1 neuron since we are predicting a single value (house price)
model.add(Dense(1))

# Compile the model with Mean Squared Error loss for regression and Adam optimizer
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model performance using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Data: {mse}')

# Predict house prices for some input data
house_example = np.array([[1.2, 0.0, 3.5, 0.0, 0.4, 5.0, 70.0, 4.0]])
house_example_scaled = scaler.transform(house_example)
predicted_price = model.predict(house_example_scaled)

print(f'Predicted House Price: {predicted_price[0][0]}')
