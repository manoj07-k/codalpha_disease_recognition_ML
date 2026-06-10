# codalpha_disease_recognition_ML

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def diabetes_prediction():

    data = pd.read_csv("diabetes.csv")

    X = data[["Glucose", "BP", "Age"]]
    y = data["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

    print("\nEnter Patient Details")

    glucose = float(input("Glucose Level: "))
    bp = float(input("Blood Pressure: "))
    age = float(input("Age: "))

    result = model.predict([[glucose, bp, age]])

    if result[0] == 1:
        print("\n⚠️ High Risk of Diabetes")
    else:
        print("\n✅ Low Risk of Diabetes")


def heart_prediction():

    data = pd.read_csv("heart.csv")

    X = data[["Age", "Cholesterol", "BP"]]
    y = data["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

    print("\nEnter Patient Details")

    age = float(input("Age: "))
    chol = float(input("Cholesterol: "))
    bp = float(input("Blood Pressure: "))

    result = model.predict([[age, chol, bp]])

    if result[0] == 1:
        print("\n⚠️ High Risk of Heart Disease")
    else:
        print("\n✅ Low Risk of Heart Disease")


def cancer_prediction():

    data = pd.read_csv("cancer.csv")

    X = data[["Radius", "Texture", "Perimeter"]]
    y = data["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)

    print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

    print("\nEnter Patient Details")

    radius = float(input("Radius: "))
    texture = float(input("Texture: "))
    perimeter = float(input("Perimeter: "))

    result = model.predict([[radius, texture, perimeter]])

    if result[0] == 1:
        print("\n⚠️ Cancer Detected")
    else:
        print("\n✅ No Cancer Detected")


while True:

    print("\n" + "=" * 45)
    print("      MULTI DISEASE PREDICTION SYSTEM")
    print("=" * 45)

    print("1. Diabetes Prediction")
    print("2. Heart Disease Prediction")
    print("3. Breast Cancer Prediction")
    print("4. Exit")

    choice = input("\nEnter Choice: ")

    if choice == "1":
        diabetes_prediction()

    elif choice == "2":
        heart_prediction()

    elif choice == "3":
        cancer_prediction()

    elif choice == "4":
        print("\nThank You!")
        break

    else:
        print("\nInvalid Choice")

  
