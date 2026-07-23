import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"

column_names = [
    "Sex", "Length", "Diameter", "Height",
    "Wholeweight", "ShuckedWeight", "VisceraWeight",
    "ShellWeight", "Rings"
]

data = pd.read_csv(url, names=column_names)

data["Sex"] = data["Sex"].map({"M": 0, "F": 1, "I": 2})

X = data.drop("Rings", axis=1)
y = data["Rings"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

sample = X_test.iloc[:5]
predictions = knn.predict(sample)

print("Sample predictions for slug ages (Rings):")
print(predictions)

print("\nActual ages:")
print(y_test.iloc[:5].values)