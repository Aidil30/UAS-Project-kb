import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def create_coffee_dataset(samples=500):
    np.random.seed(42)
    caffeine = np.random.uniform(0.8, 2.0, samples)
    acid = np.random.uniform(3.0, 5.0, samples)
    sugar = np.random.uniform(0.5, 1.5, samples)
    oil = np.random.uniform(0.3, 1.0, samples)
    protein = np.random.uniform(10, 15, samples)

    quality = [
        1 if c < 1.2 and a > 4 else 
        3 if c > 1.6 and s > 1.0 else 
        2 
        for c, a, s in zip(caffeine, acid, sugar)
    ]

    data = pd.DataFrame({
        "Caffeine": caffeine,
        "Acid": acid,
        "Sugar": sugar,
        "Oil": oil,
        "Protein": protein,
        "Quality": quality
    })

    return data

def prepare_data(data):
    X = data.drop(columns=["Quality"])
    y = data["Quality"]
    return train_test_split(X, y, test_size=0.3, random_state=42)

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    return y_pred, accuracy

def plot_results(y_test, y_pred):
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    print("Membuat dataset...")
    data = create_coffee_dataset()
    print(data.head())

    print("Membagi data...")
    X_train, X_test, y_train, y_test = prepare_data(data)

    print("Melatih model...")
    model = train_random_forest(X_train, y_train)

    print("Menguji model...")
    y_pred, accuracy = test_model(model, X_test, y_test)
    print(f"Akurasi: {accuracy:.2f}")

    print("Visualisasi hasil...")
    plot_results(y_test, y_pred)