import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target

print("Preview of the dataset:\n", df.head())
x = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse}")
print(f"R² Score: {r2:.4f}")

print("\nCoefficients:")
for name, c in zip(x.columns, model.coef_):
    print(f"{name}: {c:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Median House Values ====>")
plt.ylabel("Predicted Median House Values ====>")
plt.title("Actual vs Predicted Median House Values")
plt.grid(True)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
plt.show()

joblib.dump(model, 'Models and Datasets\\linear_regression_model.pkl')
print("\nModel has been saved!")