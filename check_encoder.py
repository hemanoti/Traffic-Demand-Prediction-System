import joblib

encoders = joblib.load("label_encoders.pkl")

print("Road Type:", encoders["road_type"].classes_)
print("Road Segment:", encoders["road_segment_id"].classes_)
print("Weather:", encoders["weather_condition"].classes_)
print("Event:", encoders["event_type"].classes_)
print("Day:", encoders["day_of_week"].classes_)
print("Traffic Density:", encoders["traffic_density"].classes_)