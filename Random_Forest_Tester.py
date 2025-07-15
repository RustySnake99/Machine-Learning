import joblib as jb
import numpy as np
from sklearn.datasets import load_iris

model = jb.load("Models and Datasets\\random_forest_model.pkl")
features = ['sepal length', 'sepal width','petal length', 'petal width']
data = load_iris()

user_input = []
for i in features:
    x = float(input(f"Enter the {i.capitalize()} (in cm): "))
    user_input.append(x)

user_input = np.array(user_input).reshape(1, -1)
result = model.predict(user_input)[0]
class_name = data.target_names[result]

print("Predicted Iris species based on user input:", class_name.capitalize())