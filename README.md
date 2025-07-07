# Quality-Prediction-in-Mining-Plants
Leveraging real-time industrial data, this project builds a powerful predictive model to forecast silica impurities in ore with precision. It empowers manufacturing plants to take smarter and faster decisions- boosting the efficiency and setting a new standard for industrial AI.

This project uses real-world industrial data from a flotation plant to build a predictive system for silica impurity levels in iron ore concentrate. Using cutting-edge Machine Learning, it enables engineers to make timely and data-driven decisions—optimizing production to reduce waste and raise the bar for smart manufacturing.

# Project Overview
In mining and industrial production, maintaining product quality is very critical. This project focuses on predicting the % of silica impurity** in ore concentrate using real-time sensor data from a flotation plant, which is a key step in iron ore processing.

# Aim
With hourly lab data and continuous sensor streams, this model aims to:
Predict silica percentage every minute
Forecast silica levels 1+ hours in advance
Evaluate predictions with and without highly correlated features like % Iron Concentrate
Empower plant engineers with predictive tools for smarter process control

# Dataset Overview
The dataset spans from March to September 2017 and contains both fast (20-sec) and slow (hourly) measurements. 
https://drive.google.com/file/d/1N80d8eTDAf1JMQXGQbHDAUaMGRyA8QG3/view?usp=sharing 

Column                                 Type	Description
date	                                 Timestamp of measurement
Iron Feed, Silica Feed	               Input ore quality before flotation
Amina Flow, Ore Pulp Flow	             Critical control variables influencing the outcome
Level, Air Flow	                       Flotation process variables
Iron Concentrate, Silica Concentrate	 Final ore quality measured in the lab (target columns)

Prediction Target: Silica Concentrate (last column)

# Modeling Approach
The model uses XGBoost Regressor, a powerful gradient boosting algorithm ideal for structured and real-world datasets.

# Pipeline Includes:
Time-based train-test split (no leakage)
Imputation for missing sensor values
Standardized feature scaling
Evaluation of model with and without % Iron Concentrate
Multi-step forecasting (1 hour into the future)

# Key Questions Addressed
Can we predict percentage of Silica Concentrate every minute? Can we forecast silica one hour or more ahead to allow early action? Can we predict silica without using percentage of Iron Concentrate? Can we identify which sensors impact silica the most?

# Visualizations
1) Actual vs Predicted % Silica Concentrate Shows model performance in tracking impurity levels over time.
2) Top 10 Feature Importances Reveals the most influential process variables affecting silica content.

# Output
The key results are shown via:
Model performance metrics (R², RMSE)
Graphs: Prediction quality, feature importance
Printed correlation analysis

# My Learnings
Gained hands-on experience with real-time industrial datasets
Learned how to engineer features for noisy and multi-rate sensor streams
Built interpretable and high-performing regression models
Understood the value of predictive maintenance in manufacturing

# License
This project is part of an academic internship and should be used for educational and non-commercial purposes only.
