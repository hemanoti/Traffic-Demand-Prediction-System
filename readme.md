# 🚦 Traffic Demand Prediction System

## 📌 Project Overview

The Traffic Demand Prediction System is a Machine Learning-based web application developed using Flask. It predicts traffic demand based on road conditions, weather conditions, and other traffic-related parameters. The application helps estimate traffic volume and can support intelligent traffic management.


## 👩‍💻 Developed By

Hema

##project demo vedio
https://drive.google.com/file/d/1ldaUsNmbvM4Lq0kCI-rzDSagYRqYu__D/view?usp=drive_link

## 🛠️ Technologies Used

- Python
- Flask
- HTML
- CSS
- Machine Learning
- LightGBM
- XGBoost
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Joblib


## 📂 Dataset

- Dataset Name: Traffic Demand Dataset
- Number of Records: 280,320
- Number of Features Used: 14

## ✨ Features

- Home Dashboard
- Traffic Demand Prediction
- Interactive Web Interface
- Machine Learning Prediction
- Traffic Analysis Graphs
- Responsive Design


## 📊 Exploratory Data Analysis (EDA)

The following visualizations were generated:

- Traffic Demand Over Time
- Correlation Heatmap
- Feature Importance
- Actual vs Predicted Graph
- Residual Plot


## 🤖 Machine Learning Models Used

- LightGBM
- XGBoost

The model with the better R² score was selected as the final prediction model.


## 📈 Model Performance

| Metric | Value |
|---------|-------|
| Model | LightGBM |
| R² Score | 0.98 |
| Train Split | 80% |
| Test Split | 20% |

> Replace the values above with your actual results if they are different.


## 📁 Project Structure

```
Traffic-Demand-Prediction-System/
│
├── app.py
├── train_model.py
├── traffic_model.pkl
├── feature_names.pkl
├── label_encoders.pkl
├── requirements.txt
├── README.md
│
├── templates/
│   ├── home.html
│   └── predict.html
│
├── static/
│   ├── style.css
│   └── images/
│
└── traffic_demand_dataset1.csv



## 📸 Project Screenshots

The dashboard includes:

- Home Dashboard
- Traffic Prediction Page
- Traffic Demand Over Time
- Correlation Heatmap
- Feature Importance
- Actual vs Predicted Graph

---
## ▶️ How to Run the Project

1. Clone the repository

```
git clone https://github.com/hemanoti/Traffic-Demand-Prediction-System.git
```

2. Install the required packages

```
pip install -r requirements.txt
```

3. Run the Flask application

```
python app.py
```

4. Open the browser and visit

```
http://127.0.0.1:5000/
```

---


## 🎥 Project Demonstration

Project Demo Video:

https://drive.google.com/file/d/1ldaUsNmbvM4Lq0kCI-rzDSagYRqYu__D/view?usp=drive_link

## 📌 Future Enhancements

- Live Traffic API Integration
- Google Maps Integration
- Real-Time Prediction
- Deep Learning Models
- Traffic Congestion Alerts

---
