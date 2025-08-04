import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
import os

warnings.filterwarnings('ignore')

EXCEL_FILE = "diabetes_model_dataset.xlsx"  # update path if needed

# Generate and save synthetic dataset
def generate_and_save_dataset(file_path):
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=1000,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                     'Insulin', 'BMI', 'FamilyHistory', 'Age']
    df = pd.DataFrame(X, columns=feature_names)
    df['Outcome'] = y
    df.to_excel(file_path, index=False)
    print(f" Dataset generated and saved to {file_path}")
    return df

# Load dataset
def load_dataset(file_path):
    df = pd.read_excel(file_path)
    print(f" Loaded dataset from {file_path}")
    return df

# Preprocess
def preprocess(df):
    for col in df.columns[:-1]:
        df[col].fillna(df[col].median(), inplace=True)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, X_test, scaler

# Train model
def train_model(X_train, y_train):
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)
    return clf

# Evaluate model
def evaluate(clf, X_test_scaled, y_test):
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return acc, cm, report, y_pred, clf.predict_proba(X_test_scaled)[:, 1]

# Predict using custom user input
def predict_user_input(clf, scaler):
    print("\n Make a prediction based on your input values")
    features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'FamilyHistory', 'Age']
    try:
        user_input = []
        for feature in features:
            val = float(input(f"Enter {feature}: "))
            user_input.append(val)

        input_array = np.array(user_input).reshape(1, -1)
        scaled_input = scaler.transform(input_array)

        label = clf.predict(scaled_input)[0]
        prob = clf.predict_proba(scaled_input)[0][1]

        print("\n Prediction Result:")
        print(f"Prediction: {'Diabetic' if label == 1 else 'Non-Diabetic'}")
        print(f"Probability of being Diabetic: {prob:.2f}")

    except ValueError:
        print(" Invalid input. Please enter numerical values only.")

# Main
def main():
    print(" Loading or creating diabetes dataset...\n")

    if os.path.exists(EXCEL_FILE):
        df = load_dataset(EXCEL_FILE)
    else:
        df = generate_and_save_dataset(EXCEL_FILE)

    X_train_scaled, X_test_scaled, y_train, y_test, X_test, scaler = preprocess(df)
    clf = train_model(X_train_scaled, y_train)
    acc, cm, report, y_test_pred, y_test_proba = evaluate(clf, X_test_scaled, y_test)

    print("\n Model Evaluation:")
    print(f" Accuracy Score: {acc:.4f}")
    print(" Confusion Matrix:")
    print(cm)
    print("\n Classification Report:")
    print(report)

    # Prediction results table
    results_df = pd.DataFrame(X_test, columns=df.columns[:-1])
    results_df['Actual Outcome'] = y_test.values
    results_df['Predicted Outcome'] = y_test_pred
    results_df['Probability Diabetic'] = y_test_proba

    print("\n Sample Predictions:")
    print(results_df.head(10))

    return clf, scaler

if __name__ == "__main__":
    clf, scaler = main()
    predict_user_input(clf, scaler)
