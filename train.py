import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

#Loading the data from the URL
url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
df = pd.read_csv(url)

# Select the features and target variable
# Feature: 100g_USD (price per 100g in USD)
# Target: Rating
X = df[['100g_USD']]
y = df['rating']

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Save the trained model as a pickle file
with open("model_1.pickle", "wb") as f:
    pickle.dump(model, f)

# Encode the categorical feature 'roast' to numeric
# Create a mapping dictionary for roast levels
roast_map = {category: code for code, category in \
            enumerate(df['roast'].unique(), start=1)}

# Apply the mapping to create a new column
df['roast_cat'] = df['roast'].map(roast_map)

# Define features (X) and target (y)
X = df[['100g_USD', 'roast_cat']]
y = df['rating']

# Train a Decision Tree Regressor
model = DecisionTreeRegressor(random_state=42)
model.fit(X, y)

# Save the model and the category mapping as model_2.pickle
with open("model_2.pickle", "wb") as f:
    pickle.dump({'model': model, 'roast_map': roast_map}, f)