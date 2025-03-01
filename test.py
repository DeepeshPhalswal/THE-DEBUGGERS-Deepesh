import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib  # To save and load the scaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from data import find_polygon_containing_point

polygons = {
    "Siberian Forest": [(60.0, 100.0), (61.0, 101.0), (62.0, 99.5), (61.5, 98.5)],
    "bwindi_impendentable" : [(32.11, 1.68),(32.25, 2.27),(31.9, 2.5),(31.6, 2.6),(31.36, 1.95),(32.0, 1.6)],
    "mangroves" : [(-81.4, 29.7),(-82.78, 28.4),(-81.0, 25.2),(-80.48, 25.33),(-80.0, 26.8)],
    "sundarbans" : [(88.9, 22.46),(89.1, 21.8),(89.9, 21.85),(91.1, 22.0),(90.9, 23.0),(90.1, 23.68),(88.8, 23.9)],
    "chamela_cuixmala_resene" : [(-103.87, 21.31),(-104.4, 20.68),(-104.62, 20.45),(-104.39, 19.92),(-103.86, 19.52),(-103.1, 19.1),(-102.23, 19.1),(-101.9, 19.4),(-102.5, 20.6),(-103.4, 21.1),(-103.9, 21.3)],
    "gir_forest" : [(70.1, 21.8),(69.9, 21.5),(70.9, 21.3),(71.29, 21.6),(70.54, 21.84)],
    "sundaland_rainforest" : [(109.282, 1.7),(109.567, -1.15),(111.237, -2.95),(115.764, -3.56),(117.912, 0.78),(117.822, 2.49),(117.982, 5.3),(116.982, 6.8),(114.882, 4.3),(113.432, 3.68),(111.682, 2.),(111.142, 1.66)],
    "amazon_rainforest" : [(-63.21, 7.9),(-72.21, 4.31),(-75.305 -0.216),(-71.22, 3.6),(-73.62, -4.4),(-71.42, -8.2),(-66.53, -15.4),(-54.58, -18.2),(-55.99, -21.9),(-60.73, -24.8),(-52.47, -27.1),(-48.25, -27.8),(-47.02, -23.),(-41.75, -12.7),(-36.82, -9.9),(-42.82, -4.7),(-54.02, 2.83),(-64.42, 8.0)],
    "jiuzhaigou_valley" : [(106.3, 30.9),(105.4, 30.8),(105.4, 30.3),(106.9, 30.40),(106.7, 30.70)],
    "greatsmoky_mountain" : [(-82.6, 38.30),(-81.9, 37.3),(-80.6, 37.3),(-79.7, 38.2),(-78.6, 38.8),(-78.2, 39.53),(-79.7, 39.7),(-80.7, 39.80),(-80.9, 39.7),(-81.8, 39.0)],
    "daintree_rainforest" : [(143.9, -14.45),(143.7, -16.29),(145.4, -17.47),(145.3, -15.62),(144.6, -14.2)],
    "black_forest" : [(8.3, 49.3),(8.5, 49.2),(7.7, 48.),(8.5, 47.6),(10.7, 49.61),(9.0, 49.7),(8.0, 48.9)],
    "alaska_boreal" : [(-159.5, 70.2),(-166.4, 68.6),(-160.7, 66.1),(-164.7, 66.2),(-163.1, 64.9),(-160.9, 63.6),(-165.1, 63.0),(-164.3, 60.5),(-155.5, 58.8),(-147.5, 61.3),(-148.4, 70.2)],
    "boreal_forest" : [(-136.68, 68.60),(-135.48, 67.28),(-132.18, 66.1),(-132.98, 65.23),(-131.36, 63.802),(-129.68, 63.01),(-126.96, 61.292),(-122.28, 59.82),(-116.18, 60.1),(-103.06, 59.911),(-102.35, 61.460),(-103.18, 64.3),(-112.08, 65.45),(-120.18, 67.7),(-124.58, 69.30),(-131.18, 69.9)],
    "siberian_taiga" : [(83.5, 62.09),(86.3, 61.25),(84.2, 60.48),(86.3, 59.8),(88.1, 59.52),(88.9, 58.2),(88.9, 55.46),(91.1, 54.55),(92.5, 52.14),(94.7, 53.210),(96.7, 55.26),(99.5, 57.78),(101.5, 58.434),(105.5, 60.44),(106.5, 62.37),(106.5, 64.27),(106.5, 65.82),(106.5, 67.22),(106.5, 68.45),(106.5, 69.24),(108.5, 70.1),(111.5, 71.06),(110.5, 72.22),(109.5, 73.1),(99.6, 72.5),(96.1, 73.1),(89.0, 73.37),(86.6, 73.01),(80.4, 71.34),(79.7, 69.8),(82.4, 67.53),(84.5, 66.0),(85.2, 64.38),(85.3, 62.98)],
    "scandinavian_taiga" : [(20.87, 68.5),(16.87, 67.9),(14.57, 66.04),(11.47, 62.501),(11.83, 59.69),(11.03, 58.61),(13.03, 56.29),(12.14, 55.511),(14.03, 55.41),(15.33, 56.3),[16.3, 57.3],(17.23, 58.38),(18.73, 59.11),(19.26, 59.607),(17.94, 60.572),(17.33, 61.4),(17.53, 62.13),(18.63, 62.86),(20.73, 63.88),(21.13, 64.87),(22.13, 65.57),(24.03, 65.89),(23.33, 66.6),(23.43, 67.53),(22.93, 68.42),(21.53, 68.70),(20.33, 69.12)]
}

# Example Prediction:
temp = input("Enter Temperature: ")
wind = input("Enter Wind Speed: ")
veg = input("Enter Vegetation Stress: ")
Humidity = input("Enter Humidity: ")
x = input("Enter X cordinate: ") 
y = input("Enter Y cordinate: ")
points = (x,y)
result = find_polygon_containing_point(points,polygons)
if(result == "alaska_boreal"):
    # Step 1: Load the trained model
    model = load_model("./model/alaska_boreal_forest_fire.h5")
    # Step 2: Load the saved scaler (if previously saved)
    scaler = joblib.load("./scaler/alaska_boreal_forest_fire.pkl")
elif(result == "Siberian Forest"):
    # Step 1: Load the trained model
    model = load_model("./model/siberian_taiga_fire.h5")
    # Step 2: Load the saved scaler (if previously saved)
    scaler = joblib.load("./scaler/siberian_taiga_fire.pkl")
elif(result == "amazon_rainforest"):
    # Step 1: Load the trained model
    model = load_model("./model/amazon_rainforest_fire.h5")
    # Step 2: Load the saved scaler (if previously saved)
    scaler = joblib.load("./scaler/amazon_rainforest_fire.pkl")
elif(result == "bwindi_impendentable"):
    # Step 1: Load the trained model
    model = load_model("./model/bwindi_forest_fire.h5")
    # Step 2: Load the saved scaler (if previously saved)
    scaler = joblib.load("./scaler/bwindi_forest_fire.pkl")
elif(result == "mangroves"):
    # Step 1: Load the trained model
    model = load_model("./model/everglades_mangroves_fire.h5")
    # Step 2: Load the saved scaler (if previously saved)
    scaler = joblib.load("./scaler/everglades_mangroves_fire.pkl")
elif(result == "sundarbans"):
    # Step 1: Load the trained model
    model = load_model("./model/sundarbans_forest_fire.h5")
    # Step 2: Load the saved scaler (if previously saved)
    scaler = joblib.load("./scaler/sundarbans_forest_fire.pkl")
elif(result == "chamela_cuixmala_resene"):
    # Step 1: Load the trained model
    model = load_model("./model/chamela_cuixmala_fire.h5")
    # Step 2: Load the saved scaler (if previously saved)
    scaler = joblib.load("./scaler/chamela_cuixmala_fire.pkl")
elif(result == "gir_forest"):
    # Step 1: Load the trained model
    model = load_model("./model/gir_forest_fire.h5")
    # Step 2: Load the saved scaler (if previously saved)
    scaler = joblib.load("./scaler/gir_forest_fire.pkl")
elif(result == "sundaland_rainforest"):
    # Step 1: Load the trained model
    model = load_model("./model/sunderland_forest_fire.h5")
    # Step 2: Load the saved scaler (if previously saved)
    scaler = joblib.load("./scaler/sunderland_forest_fire.pkl")
elif(result == "jiuzhaigou_valley"):
    # Step 1: Load the trained model
    model = load_model("./model/jiuzhaigou_valley_fire.h5")
    # Step 2: Load the saved scaler (if previously saved)
    scaler = joblib.load("./scaler/jiuzhaigou_valley_fire.pkl")
elif(result == "greatsmoky_mountain"):
    # Step 1: Load the trained model
    model = load_model("./model/great_smoky_mountain_fire.h5")
    # Step 2: Load the saved scaler (if previously saved)
    scaler = joblib.load("./scaler/great_smoky_mountain_fire.pkl")
elif(result == "daintree_rainforest"):
    # Step 1: Load the trained model
    model = load_model("./model/daintree_forest_fire.h5")
    # Step 2: Load the saved scaler (if previously saved)
    scaler = joblib.load("./scaler/daintree_forest_fire.pkl")
elif(result == "black_forest"):
    # Step 1: Load the trained model
    model = load_model("./model/black_forest_fire.h5")
    # Step 2: Load the saved scaler (if previously saved)
    scaler = joblib.load("./scaler/black_forest_fire.pkl")
elif(result == "boreal_forest"):
    # Step 1: Load the trained model
    model = load_model("./model/boreal_forest_fire.h5")
    # Step 2: Load the saved scaler (if previously saved)
    scaler = joblib.load("./scaler/boreal_forest_fire.pkl")
elif(result == "siberian_taiga"):
    # Step 1: Load the trained model
    model = load_model("./model/siberian_taiga_fire.h5")
    # Step 2: Load the saved scaler (if previously saved)
    scaler = joblib.load("./scaler/siberian_taiga_fire.pkl")
elif(result == "scandinavian_taiga"):
    # Step 1: Load the trained model
    model = load_model("./model/scandinavian_taiga_fire.h5")
    # Step 2: Load the saved scaler (if previously saved)
    scaler = joblib.load("./scaler/scandinavian_taiga_fire.pkl")
else:
    print("Wrong cordinates Entered")

# Step 3: Function to make predictions
def predict_fire_risk(temp, humidity, veg_index):
    new_data = np.array([[temp, humidity, veg_index]])
    new_data_scaled = scaler.transform(new_data)  # Normalize input
    fire_probability = model.predict(new_data_scaled)[0][0]  # Get probability
    
    print(f"🔥 Fire Probability: {fire_probability * 100:.2f}%")
    return "🔥 Fire Danger!" if fire_probability > 0.5 else "✅ No Fire Risk."


print(predict_fire_risk(temp,Humidity,veg))  # Replace with real input values
