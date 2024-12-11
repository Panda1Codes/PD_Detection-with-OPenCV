import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from skimage import feature


def preprocess_image(image, resize_dim=(200, 200)):
    # Convert to grayscale, resize, and apply thresholding
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, resize_dim)
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return image


def extract_features(image):
    # Compute HOG features
    return feature.hog(image, orientations=9, 
                       pixels_per_cell=(10, 10), 
                       cells_per_block=(2, 2), 
                       transform_sqrt=True, block_norm="L1")


def train_model(trainingPaths):
    data, labels = [], []
    for path in trainingPaths:
        print(f"[INFO] Loading data from {path}...")
        imagePaths = [os.path.join(path, subdir, img) 
                      for subdir in ["healthy", "parkinson"]
                      for img in os.listdir(os.path.join(path, subdir))]
        
        for imgPath in imagePaths:
            label = os.path.basename(os.path.dirname(imgPath))
            image = cv2.imread(imgPath)
            if image is None:
                print(f"[WARNING] Could not read image {imgPath}. Skipping.")
                continue
            preprocessed = preprocess_image(image)
            features = extract_features(preprocessed)
            data.append(features)
            labels.append(label)

    # Encode labels
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # Train Random Forest Classifier
    print("[INFO] Training model...")
    model = RandomForestClassifier(n_estimators=100)
    model.fit(data, labels)
    return model, le


def classify_image(model, label_encoder, image):
    preprocessed = preprocess_image(image)
    features = extract_features(preprocessed)
    prediction = model.predict([features])[0]
    label = label_encoder.inverse_transform([prediction])[0]
    return label


def main():
    dataset_path = input("[INPUT] Enter dataset path for training: ").strip()

    # Paths for spiral and wave categories
    trainingPaths = [os.path.join(dataset_path, "drawings", cat, "training") for cat in ["spiral", "wave"]]

    # Train the model
    model, label_encoder = train_model(trainingPaths)

    # Start real-time video capture
    print("[INFO] Starting real-time detection...")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Unable to capture frame. Exiting...")
            break

        # Display live camera feed
        frame_copy = frame.copy()
        cv2.putText(frame_copy, "Show a spiral or wave drawing to classify", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Real-Time Detection", frame_copy)

        # Press 's' to capture and classify the current frame
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            # Preprocess and classify the image
            detected_label = classify_image(model, label_encoder, frame)
            print(f"[INFO] Detected: {detected_label}")
            cv2.putText(frame, f"Detected: {detected_label}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Real-Time Detection", frame)

        # Press 'q' to quit
        if key == ord("q"):
            print("[INFO] Exiting...")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
