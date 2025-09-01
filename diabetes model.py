import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
#Loading the dataset 
diabetes_dataset = pd.read_csv('diabetes.csv')
print(diabetes_dataset.head())
print(diabetes_dataset.shape)
print(diabetes_dataset.describe())
diabetes_dataset.groupby('Outcome').mean()
# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']
scaler = StandardScaler()
scaler.fit(X)
StandardScaler(copy=True, with_mean=True, with_std=True)
standardized_data = scaler.transform(X)
X = standardized_data
Y = diabetes_dataset['Outcome']
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)
pregnancies = float(input("Enter number of Pregnancies: "))
glucose = float(input("Enter Glucose level: "))
blood_pressure = float(input("Enter Blood Pressure value: "))
skin_thickness = float(input("Enter Skin Thickness value: "))
insulin = float(input("Enter Insulin level: "))
bmi = float(input("Enter BMI value: "))
dpf = float(input("Enter Diabetes Pedigree Function value: "))
age = float(input("Enter Age: "))

input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
input_array = np.asarray(input_data).reshape(1, -1)
input_std = scaler.transform(input_array)
prediction = classifier.predict(input_std)

if prediction[0] == 0:
    print("The person is not diabetic.")
else:
    print("The person is diabetic.")
