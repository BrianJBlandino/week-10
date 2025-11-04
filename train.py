import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import pickle

# Loading the data
url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
df = pd.read_csv(url)

# Encode the categorical feature 'roast' to numeric
# Create a mapping (dictionary) for roast levels
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
    pickle.dump(model, f)