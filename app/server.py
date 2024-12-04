from fastapi import FastAPI
import joblib
import numpy as np

# loading pre-trained model
model = joblib.load('model.joblib')
# define classnames
class_names = np.array(['setosa', 'versicolor', 'virginica'])

app = FastAPI()

@app.get('/')
def reed_root():
    return {'message': 'Iris Model API'}

@app.post('/predict')
def predict(data : dict):
    # convert imput features into a numpy array and reshape prediction
    features = np.array(data['features'].reshape(1, -1))
    # use loaded model to predict the class of the input features
    prediction = model.predict(features)
    # maps the predicted class index to the corresponding class name
    class_name = class_names[prediction][0]
    return {'predicted_class': class_name}