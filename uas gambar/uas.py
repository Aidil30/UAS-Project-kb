import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from skimage.feature import hog
import matplotlib.pyplot as plt

def load_signature_images(data_dir, img_size=(128, 128)):
    X = []
    y = []
    for class_id, folder_name in enumerate(os.listdir(data_dir)):  
        person_dir = os.path.join(data_dir, folder_name)
        if not os.path.isdir(person_dir):  
            continue
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
            if img is None:
                print(f"Warning: Gagal membaca {img_path}")
                continue
            img_resized = cv2.resize(img, img_size)  
            X.append(img_resized)
            y.append(class_id)  
    return np.array(X), np.array(y)

def extract_hog_features(images):
    hog_features = []
    for img in images:
        features = hog(
            img, 
            orientations=9, 
            pixels_per_cell=(8, 8), 
            cells_per_block=(2, 2), 
            block_norm="L2-Hys"
        )
        hog_features.append(features)
    return np.array(hog_features)

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def test_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print("Classification Report:")
    print(classification_report(y_val, y_pred))
    return y_pred, accuracy

def plot_results(y_val, y_pred):
    disp = ConfusionMatrixDisplay.from_predictions(y_val, y_pred, cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    data_dir = "D:/uasprojekkbgambar/tanda tangan"  
    if not os.path.exists(data_dir):
        print(f"Error: Folder dataset tidak ditemukan di {data_dir}.")
        exit()

    print("Membaca dataset...")
    X, y = load_signature_images(data_dir)
    print(f"Jumlah gambar yang dibaca: {len(X)}")

    print("Ekstraksi fitur...")
    X_hog = extract_hog_features(X)
    print(f"Jumlah fitur yang diekstraksi: {len(X_hog)}")

    print("Membagi data...")
    X_train, X_val, y_train, y_val = train_test_split(X_hog, y, test_size=0.3, random_state=42)
    print(f"Jumlah data latih: {len(X_train)}, Jumlah data validasi: {len(X_val)}")

    print("Melatih model...")
    model = train_model(X_train, y_train)
    print("Model berhasil dilatih.")

    print("Menguji model...")
    y_pred, accuracy = test_model(model, X_val, y_val)
    print(f"Akurasi Validasi: {accuracy:.2f}")

    print("Visualisasi hasil...")
    plot_results(y_val, y_pred)
