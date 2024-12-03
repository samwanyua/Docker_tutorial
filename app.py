import joblib  # for saving and loading python objects efficiently
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# load iris dataset
iris = load_iris()
# split the data into features and target variables
X, y = iris.data, iris.target

# train random forest classifier
model = RandomForestClassifier()
model.fit(X, y)


# save the trained model
joblib.dump(model, 'model.joblib')