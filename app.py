from flask import Flask, render_template, request
import pandas as pd
import joblib
from datetime import datetime

app = Flask(__name__)

# Load model
model = joblib.load("traffic_model.pkl")

# Load feature names
feature_names = joblib.load("feature_names.pkl")

# Load encoders
encoders = joblib.load("label_encoders.pkl")
# Load dataset
dataset = pd.read_csv("traffic_demand_dataset1.csv")

# Total records
dataset_size = len(dataset)

# Number of features
features_count = len(feature_names)

# Model name
model_name = "LightGBM"

# Accuracy
accuracy = "98%"
@app.route("/")
def home():

    return render_template(

        "home.html",

        dataset_size=dataset_size,

        accuracy=accuracy,

        model_name=model_name,

        features_count=features_count

    )
@app.route("/predict_page")
def predict_page():
    return render_template("predict.html")
@app.route("/predict", methods=["POST"])
def predict():
        # Get user input
    timestamp = request.form["timestamp"]
    road_type = request.form["road_type"]
    number_of_lanes = int(request.form["number_of_lanes"])
    speed_limit = int(request.form["speed_limit"])
    weather_condition = request.form["weather_condition"]
    temperature = float(request.form["temperature"])
    humidity = int(request.form["humidity"])
    rainfall = float(request.form["rainfall"])
    wind_speed = float(request.form["wind_speed"])
    visibility = int(request.form["visibility"])
    nearby_intersections = int(request.form["nearby_intersections"])
    event_type = request.form["event_type"]    
        # Convert timestamp
    dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M")

    hour = dt.hour
    day_of_week = dt.strftime("%A")

    # Peak hour
    if hour in [7, 8, 9, 17, 18, 19]:
        peak_hour = 1
    else:
        peak_hour = 0
        # Encode categorical values
    road_type = encoders["road_type"].transform([road_type])[0]
    day_of_week = encoders["day_of_week"].transform([day_of_week])[0]
    weather_condition = encoders["weather_condition"].transform([weather_condition])[0]
    event_type = encoders["event_type"].transform([event_type])[0]    
        # Create DataFrame for prediction
    data = pd.DataFrame([{
        "road_type": road_type,
        "number_of_lanes": number_of_lanes,
        "speed_limit": speed_limit,
        "day_of_week": day_of_week,
        "hour": hour,
        "weather_condition": weather_condition,
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
        "wind_speed": wind_speed,
        "visibility": visibility,
        "nearby_intersections": nearby_intersections,
        "event_type": event_type,
        "peak_hour": peak_hour
    }])

    data = data[feature_names]

    print(data)
    # Predict
    prediction = model.predict(data)[0]
    return render_template(
    "predict.html",
    prediction_text=f"Predicted Traffic Demand: {prediction:.2f}"
)
if __name__ == "__main__":
    app.run(debug=True)