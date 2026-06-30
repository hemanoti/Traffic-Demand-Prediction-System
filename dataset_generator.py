import pandas as pd
import numpy as np

np.random.seed(42)

# -------------------------------
# CONFIGURATION
# -------------------------------
# -------------------------------
# CONFIGURATION
# -------------------------------

NUM_ROADS = 4   # 4 road segments

# Generate timestamps for exactly 2 years
timestamps = pd.date_range(
    start="2022-01-01 00:00:00",
    end="2023-12-31 23:45:00",
    freq="15min"
)

road_ids = [f"R{str(i).zfill(3)}" for i in range(1, NUM_ROADS + 1)]

# Create one record for every timestamp and road
df = pd.MultiIndex.from_product(
    [timestamps, road_ids],
    names=["timestamp", "road_segment_id"]
).to_frame(index=False)

NUM_ROWS = len(df)

print("Total Rows:", NUM_ROWS)

road_ids = [f"R{str(i).zfill(3)}" for i in range(1, NUM_ROADS + 1)]

road_types = ["Urban", "Highway", "Rural"]
weather_conditions = ["Clear", "Cloudy", "Rain", "Fog"]
event_types = ["None", "Festival", "Concert", "Sports", "Accident"]

# -------------------------------
# CREATE BASE DATA
# -------------------------------

# Coordinates
df["latitude"] = np.random.uniform(12.80, 13.20, NUM_ROWS)
df["longitude"] = np.random.uniform(77.40, 77.80, NUM_ROWS)

df["road_type"] = np.random.choice(
    road_types,
    NUM_ROWS,
    p=[0.6,0.25,0.15]
)

df["number_of_lanes"] = np.random.choice(
    [2,4,6,8],
    NUM_ROWS,
    p=[0.3,0.4,0.2,0.1]
)

df["speed_limit"] = np.where(
    df["road_type"]=="Highway",
    np.random.choice([80,100,120],NUM_ROWS),
    np.where(
        df["road_type"]=="Urban",
        np.random.choice([40,50,60],NUM_ROWS),
        np.random.choice([30,40,50],NUM_ROWS)
    )
)

# -------------------------------
# TIME FEATURES
# -------------------------------

df["day_of_week"] = df.timestamp.dt.day_name()
df["month"] = df.timestamp.dt.month
df["hour"] = df.timestamp.dt.hour
df["minute"] = df.timestamp.dt.minute

df["is_weekend"] = df.timestamp.dt.dayofweek >= 5

# Random holidays
df["is_holiday"] = np.random.choice(
    [0,1],
    NUM_ROWS,
    p=[0.95,0.05]
)

# -------------------------------
# WEATHER
# -------------------------------

df["weather_condition"] = np.random.choice(
    weather_conditions,
    NUM_ROWS,
    p=[0.55,0.20,0.20,0.05]
)

df["temperature"] = np.random.normal(28,5,NUM_ROWS).round(1)

df["humidity"] = np.random.randint(40,95,NUM_ROWS)

df["rainfall"] = np.where(
    df.weather_condition=="Rain",
    np.random.uniform(1,15,NUM_ROWS),
    0
).round(2)

df["wind_speed"] = np.random.uniform(2,30,NUM_ROWS).round(1)

df["visibility"] = np.where(
    df.weather_condition=="Fog",
    np.random.randint(200,1000,NUM_ROWS),
    np.random.randint(3000,10000,NUM_ROWS)
)

# -------------------------------
# ROAD FEATURES
# -------------------------------

df["nearby_intersections"] = np.random.randint(0,8,NUM_ROWS)

df["historical_avg_speed"] = np.random.uniform(20,80,NUM_ROWS).round(1)

df["event_type"] = np.random.choice(
    event_types,
    NUM_ROWS,
    p=[0.90,0.03,0.03,0.02,0.02]
)

# Peak hour
df["peak_hour"] = df.hour.isin([7,8,9,17,18,19]).astype(int)

# -------------------------------
# WEATHER SCORE
# -------------------------------

weather_score = []

for w in df.weather_condition:

    if w=="Clear":
        weather_score.append(1.0)

    elif w=="Cloudy":
        weather_score.append(0.8)

    elif w=="Rain":
        weather_score.append(0.5)

    else:
        weather_score.append(0.3)

df["weather_score"] = weather_score

# -------------------------------
# EVENT SCORE
# -------------------------------

event_score = []

for e in df.event_type:

    if e=="None":
        event_score.append(0)

    elif e=="Festival":
        event_score.append(40)

    elif e=="Concert":
        event_score.append(35)

    elif e=="Sports":
        event_score.append(45)

    else:
        event_score.append(60)

# -------------------------------
# TARGET VARIABLE
# -------------------------------

base = 40

traffic = (
    base
    + df.number_of_lanes*18
    + df.peak_hour*90
    + df.nearby_intersections*8
    + np.array(event_score)
    + np.random.normal(0,12,NUM_ROWS)
)

traffic += np.where(df.weather_condition=="Rain",-15,0)
traffic += np.where(df.weather_condition=="Fog",-20,0)
traffic += np.where(df.road_type=="Highway",30,0)
traffic += np.where(df.is_weekend,-20,0)
traffic += np.where(df.is_holiday,-30,0)

traffic = np.maximum(10, traffic)

df["traffic_demand"] = traffic.astype(int)

# -------------------------------
# CONGESTION INDEX
# -------------------------------

df["congestion_index"] = (
    df.traffic_demand /
    df.traffic_demand.max()
).round(3)

# -------------------------------
# TRAFFIC DENSITY
# -------------------------------

conditions = [
    df.traffic_demand < 80,
    (df.traffic_demand >=80) &
    (df.traffic_demand <160),
    df.traffic_demand >=160
]

choices = [
    "Low",
    "Medium",
    "High"
]

df["traffic_density"] = np.select(
    conditions,
    choices,
    default="Medium"
)

# -------------------------------
# LAG FEATURES
# -------------------------------

df = df.sort_values(["road_segment_id","timestamp"])

df["lag_1"] = df.groupby("road_segment_id")["traffic_demand"].shift(1)
df["lag_2"] = df.groupby("road_segment_id")["traffic_demand"].shift(2)
df["lag_3"] = df.groupby("road_segment_id")["traffic_demand"].shift(3)

df["rolling_mean_4"] = (
    df.groupby("road_segment_id")["traffic_demand"]
      .transform(lambda x: x.rolling(4,min_periods=1).mean())
)

df = df.bfill()

# -------------------------------
# SAVE
# -------------------------------

df.to_csv(
    "traffic_demand_dataset1.csv",
    index=False
)

print(df.head())
print(df.shape)
print("\nDataset Saved Successfully!")