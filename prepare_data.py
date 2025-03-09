import pandas as pd

# Load dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Select relevant features
df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})  # Convert gender to numeric
df = df.dropna()  # Remove missing values

# Save dataset
df.to_csv('titanic.csv', index=False)
print("Dataset saved as 'titanic.csv'")
