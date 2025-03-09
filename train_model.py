import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the Titanic dataset (update the path as needed)
df = pd.read_csv("titanic.csv")  

# Preprocess the dataset (modify based on your dataset)
df = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Survived"]]
df.dropna(inplace=True)
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})  # Convert 'Sex' to numeric

X = df.drop(columns=["Survived"])  # Features
y = df["Survived"]  # Target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model correctly
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model retrained and saved as model.pkl")
