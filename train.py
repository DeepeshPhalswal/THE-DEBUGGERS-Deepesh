import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import LeakyReLU
import joblib

# Step 1: Load the data
data = pd.read_csv("./data/train/sundaland_rainforest_fire_data.csv")

# Step 2: Split features and target
X = data[['Temperature (°C)', 'Humidity (%)', 'Vegetation Index']]
y = data['Fire Occurrence']

# Step 3: Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the correct scaler
joblib.dump(scaler, "./scaler/sundaland_rainforest_fire.pkl")
print("✅ Correct scaler.pkl saved.")

# Step 4: Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
 
# Step 5: Build the deep learning model (Random Forest-like DNN)
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Increased neurons
    BatchNormalization(),  # Normalize inputs
    Dropout(0.2),  # Prevent overfitting
    
    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(64),
    LeakyReLU(alpha=0.1),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Step 6: Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 7: Train the model for 5 epochs
history = model.fit(X_train, y_train, epochs=50, batch_size=2, validation_data=(X_test, y_test), verbose=1)

# Step 8: Evaluate the model on test data
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probability to binary output

# Print model evaluation metrics
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 9: Save the trained model
model.save("./model/sundaland_rainforest_fire.h5")  # Save the model in the current directory


# Step 10: Predict fire risk for a new input
def predict_fire_risk(temp, humidity, veg_index):
    new_data = np.array([[temp, humidity, veg_index]])
    new_data_scaled = scaler.transform(new_data)
    fire_probability = model.predict(new_data_scaled)[0][0]
    
    print(f"🔥 Fire Probability: {fire_probability * 100:.2f}%")
    return "🔥 Fire Danger!" if fire_probability > 0.5 else "✅ No Fire Risk."

# Example Prediction:
print(predict_fire_risk(10, 40, 0.7))  # Example input


# Step 11: Plot Training vs Validation Loss & Accuracy
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()

plt.show()
