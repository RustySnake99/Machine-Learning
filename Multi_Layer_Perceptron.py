from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

digits = load_digits()
x, y = digits.data, digits.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=1000, alpha=1e-4, solver='adam', random_state=1, verbose=True)
model.fit(x_train, y_train)

y_prediction = model.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_prediction) * 100:.2f}%")
print(f"Classification Report:\n{classification_report(y_test, y_prediction)}")
joblib.dump(model, "Models and Datasets/multilayer_perceptron_sklearn.pkl")
print("Model has been successfully saved!")