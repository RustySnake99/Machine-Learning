from sklearn.model_selection import KFold, cross_val_score
from sklearn.datasets import load_wine
from sklearn.svm import SVC

data = load_wine()
x, y = data.data, data.target
model = SVC()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, x, y, cv=kf)
print("Cross Validation Scores (Accuracy Scores, in percentage):", [f"{(i * 100):.2f}%" for i in scores])
print(f"Average Accuracy: {(scores.mean() * 100):.2f}%")