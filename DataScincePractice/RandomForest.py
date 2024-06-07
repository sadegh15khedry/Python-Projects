from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import pandas as pd

# Load a sample dataset (replace with your actual data)
df = pd.read_csv(r"dataset.csv")
x = X = df.drop('Outcome', axis=1)
y = df["Outcome"]


# Train the model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x, y)

# Print feature importances
print(clf.feature_importances_)