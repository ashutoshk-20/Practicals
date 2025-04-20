import joblib
from sklearn.datasets import load_iris

iris = load_iris()


loaded_model = joblib.load('model.joblib')
loaded_scaler = joblib.load('scaler.joblib')

sample = [[5.1,3.5,1.4,0.2]]
scaled_sample = loaded_scaler.transform(sample)
prediction = loaded_model.predict(scaled_sample)

print(f"Prediction: {iris.target_names[prediction[0]]}")