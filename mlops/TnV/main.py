import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

os.makedirs('artifacts',exist_ok=True)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Iris Classification")

iris = load_iris()
X=iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models_config = {
    "Logistic Regression":{
        "model": LogisticRegression(max_iter=200),
        "params":{
            'C':[0.1,1.0,10],
        }
    },
    "Random Forest":{
        "model": RandomForestClassifier(),
        "params":{
            'n_estimators':[50,100],
            'max_depth':[2,None],
        }
    },
    "SVM":{
        "model": SVC(),
        "params":{
            'C':[1.0,10],
            'kernel':['linear','rbf'],
        }
    }
}

for name,config in models_config.items():
    clf = GridSearchCV(config['model'],config['params'],cv=5)
    clf.fit(X_train,y_train)
    best_model = clf.best_estimator_
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    cm = confusion_matrix(y_test,y_pred)

    fig,ax = plt.subplots(figsize=(8,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap="Blues",ax=ax,xticklabels=iris.target_names,yticklabels=iris.target_names)
    plt.title(f'Confusion Matrix for {name}')
    cm_path = f"artifacts/{name}_confusion_matrix.png"
    fig.savefig(cm_path)
    plt.close(fig)

    with mlflow.start_run(run_name=name):
        mlflow.sklearn.log_model(best_model,"model")
        mlflow.log_params(clf.best_params_)
        mlflow.log_param("model_name",name)

        mlflow.log_metric('accuracy',acc)
        mlflow.log_artifact(cm_path)
        print(f"Model: {name}, Accuracy: {acc}")
