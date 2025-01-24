This project demonstrates a workflow for predicting student performance based on various features using the Random Forest Regressor algorithm. The dataset used includes attributes that influence student performance, and the model aims to predict the final grades.

Features:- Data Loading: Loading the student performace data from an excel file. Preprocessing:Categorical features are encoded using LabelEncoder, and a correlation heatmap is generated.
Feature Engineering: Features are prepared for model training by splitting into training and testing sets and standardizing the numerical features.
Model Training: A Random Forest Regressor model is trained on the data. Model Evaluation: The model's performance is evaluated using Mean Squared Error (MSE) and R-squared score.
A scatter plot of actual vs. predicted values is also shown. Feature Importance: The importance of each feature is calculated and visualized. Model Saving: The trained model is saved to a file for future use.

Dataset:- Created a dataset of Student performance.

Step by step:-

Load the Dataset:
Fetches the dataset of Student performance. Checks for missing values.

Preprocessing:
Splits the data into features (X) and target (y). Divides the data into training and testing sets. Scales the features using StandardScaler.

Feature Engineering:
Features are prepared for model training by splitting into training and testing sets and standardizing the numerical features.

Model Training:
Trains a RandomForestRegressor on the scaled training data.

Evaluation:
Predicts grades for the student's performance. Evaluates the model using: Mean Absolute Error (MAE) Mean Squared Error (MSE) Root Mean Squared Error (RMSE) R-squared (R²)

Visualization:
Plots the true vs. predicted prices. Displays the feature importance as a bar chart. Results The script outputs:

Performance metrics (MAE, MSE, RMSE, R²). Visualizations: Scatter plot comparing true and predicted housing prices. Bar plot of feature importance.
