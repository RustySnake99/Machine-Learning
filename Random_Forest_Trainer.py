from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib as jb
import matplotlib.pyplot as plt

data = load_iris()
x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(x_train, y_train)

importance = model.feature_importances_
features = data.feature_names

plt.barh(features, importance)
plt.xlabel("Feature importance")
plt.title("Random Forest Feature-Importances")
plt.show()

y_prediction = model.predict(x_test)
print("Accuracy: ", accuracy_score(y_test, y_prediction))
print("Classification Report:\n",classification_report(y_test, y_prediction, target_names=data.target_names))

jb.dump(model, "Models and Datasets\\random_forest_model.pkl")
print("Model has been successfully saved!!")