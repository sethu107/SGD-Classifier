# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
## Step 1:
Import Necessary Libraries and Load Data.
## Step 2:
Split Dataset into Training and Testing Sets.
## Step 3:
Train the Model Using Stochastic Gradient Descent (SGD).
## Step 4:
Make Predictions and Evaluate Accuracy.
## Step 5:
Generate Confusion Matrix.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Sethupathi K
RegisterNumber: 212223040189

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

#load the iris dataset
iris = load_iris()

#create a pandas dataframe
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']=iris.target

#print the first 5 values
print(df.head())

#split the data into features (x) and(y)
X=df.drop('target',axis=1)
Y=df['target']

#split the data into training and testing sets
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2,random_state=42)

#create an SGD classifier with default parameters
sgd_clf=SGDClassifier(max_iter=1000,tol=1e-3)

#train the classifier on thr training data
sgd_clf.fit(X_train,Y_train)

#make predictions on the testing data
y_pred=sgd_clf.predict(X_test)

print(f"Accuracy:{accuracy:.3f}")

#calculate the confusion matrix
cf=confusion_matrix(Y_test, y_pred)
print("Confusion Matrix")
print(cf)
*/
```

## Output:
![Screenshot 2024-09-19 092403](https://github.com/user-attachments/assets/57754234-fde5-404c-b4d6-ad7d1e2428bb)
![Screenshot 2024-09-19 092411](https://github.com/user-attachments/assets/7b222ca5-1dfe-4a2c-b7fa-608592974bc7)
![Screenshot 2024-09-19 092416](https://github.com/user-attachments/assets/ca871ec6-4b15-454d-a65b-239c642e2b1a)

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
