import numpy as np
import pandas as pd

import os
import warnings
warnings.simplefilter(action="ignore")

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
movies_data = pd.read_csv('/content/imdb-movies-dataset.csv')

# Drop irrelevant columns
cleaned_data = movies_data.drop(['Poster', 'Description', 'Title', 'Metascore', 'Review Count', 'Review Title', 'Review'], axis=1)
cleaned_data['Votes'] = cleaned_data['Votes'].str.replace(',', '').astype(float)

# Fill missing values
cleaned_data.fillna({
    'Year': cleaned_data['Year'].mean(),
    'Certificate': cleaned_data['Certificate'].mode()[0],
    'Duration (min)': cleaned_data['Duration (min)'].mean(),
    'Genre': cleaned_data['Genre'].mode()[0],
    'Rating': cleaned_data['Rating'].mean(),
    'Director': cleaned_data['Director'].mode()[0],
    'Cast': cleaned_data['Cast'].mode()[0],
    'Votes': cleaned_data['Votes'].mean(),
}, inplace=True)

# Explode genres
cleaned_data['Genre'] = cleaned_data['Genre'].str.split(', ')
cleaned_data = cleaned_data.explode('Genre')

# Encode categorical features
cleaned_data['encoded_Genre'] = cleaned_data.groupby('Genre')['Rating'].transform('mean')
cleaned_data['encoded_Duration'] = cleaned_data.groupby('Duration (min)')['Rating'].transform('mean')
cleaned_data['encoded_Certificate'] = cleaned_data.groupby('Certificate')['Rating'].transform('mean')
cleaned_data['encoded_Director'] = cleaned_data.groupby('Director')['Rating'].transform('mean')

# Define features and target
X = cleaned_data[['Votes', 'encoded_Certificate', 'encoded_Duration', 'encoded_Genre', 'encoded_Director']]
y = cleaned_data['Rating']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train models
svr = SVR(kernel='rbf')
svr.fit(X_train, y_train)

dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# Evaluate models
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2s = r2_score(y_test, predictions)
    return mse, r2s

svr_mse, svr_r2s = evaluate_model(svr, X_test, y_test)
dt_mse, dt_r2s = evaluate_model(dt, X_test, y_test)
rf_mse, rf_r2s = evaluate_model(rf, X_test, y_test)

# Print model performance
print(f"SVR Mean squared error: {svr_mse}, R2 score: {svr_r2s}")
print(f"DT Mean squared error: {dt_mse}, R2 score: {dt_r2s}")
print(f"RF Mean squared error: {rf_mse}, R2 score: {rf_r2s}")

# Function to encode new data
def encode_new_data(test_data, cleaned_data):
    test_data["encoded_Certificate"] = cleaned_data[cleaned_data['Certificate'] == test_data['Certificate']]['Rating'].iloc[0]
    test_data["encoded_Duration"] = cleaned_data[cleaned_data['Duration (min)'] == test_data['Duration (min)']]['Rating'].iloc[0]
    test_data["encoded_Genre"] = cleaned_data[cleaned_data['Genre'] == test_data['Genre']]['Rating'].iloc[0]
    test_data['encoded_Director'] = cleaned_data[cleaned_data['Director'] == test_data['Director']]['Rating'].iloc[0]
    return pd.DataFrame(test_data, index=[0]).drop(['Certificate', 'Duration (min)', 'Genre', 'Director'], axis=1)

# Get user input
user_certificate = input("Enter Certificate (e.g., PG-13): ")
user_votes = int(input("Enter Votes: "))
user_duration = float(input("Enter Duration (min): "))
user_genre = input("Enter Genre: ")
user_director = input("Enter Director: ")

# Prepare test data
test_data = {
    'Certificate': user_certificate,
    'Votes': user_votes,
    'Duration (min)': user_duration,
    'Genre': user_genre,
    'Director': user_director
}

encoded_test_data = encode_new_data(test_data, cleaned_data)

# Make predictions
predicted_rating_svr = svr.predict(encoded_test_data)
predicted_rating_dt = dt.predict(encoded_test_data)
predicted_rating_rf = rf.predict(encoded_test_data)

# Print predictions
print(f"Predicted Rating by SVR: {predicted_rating_svr[0]}")
print(f"Predicted Rating by Decision Tree: {predicted_rating_dt[0]}")
print(f"Predicted Rating by Random Forest: {predicted_rating_rf[0]}")
