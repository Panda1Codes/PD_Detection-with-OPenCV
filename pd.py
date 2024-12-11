# Command Line Argument : python script.py --dataset /path/to/dataset
import os
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from skimage import feature
from imutils import build_montages
import numpy as np
import cv2
import argparse


def list_images(directory):
    # Include common image extensions explicitly
    valid_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    imagePaths = []
    for ext in valid_extensions:
        imagePaths.extend(glob.glob(os.path.join(directory, "**", ext), recursive=True))
    return imagePaths

#Preprocessing (OpenCV)
def preprocess_image(image, resize_dim=(200, 200)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, resize_dim)
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return image

#FEature Extraction using HOG
def extract_features(image):
    return feature.hog(image, orientations=9, 
                       pixels_per_cell=(10, 10), 
                       cells_per_block=(2, 2), 
                       transform_sqrt=True, block_norm="L1")

#Loading dataset
def load_data(data_path, resize_dim=(200, 200)):
    print(f"[INFO] Loading data from {data_path}...")
    imagePaths = list_images(data_path)
    print(f"[DEBUG] Found {len(imagePaths)} images in {data_path}.")
    if not imagePaths:
        raise ValueError(f"No images found in path: {data_path}")

    data, labels = [], []
    for path in imagePaths:
        # Parse the class label (e.g., 'healthy' or 'parkinson') from directory structure
        label = os.path.basename(os.path.dirname(path))
        if label not in ["healthy", "parkinson"]:  # Update for your specific dataset categories
            print(f"[WARNING] Skipping unknown category {label} in {path}")
            continue
        image = cv2.imread(path)
        if image is None:
            print(f"[WARNING] Could not read image {path}. Skipping.")
            continue
        preprocessed = preprocess_image(image, resize_dim)
        features = extract_features(preprocessed)
        data.append(features)
        labels.append(label)

    if not data or not labels:
        raise ValueError(f"Data loading failed. No valid images in {data_path}.")
    return np.array(data), np.array(labels)

#Evaluation (Metrics used are : Accuracy, Sensitivy, Specificity)
def evaluate_model(model, testX, testY, label_encoder):
    print("[INFO] Evaluating the model...")
    predictions = model.predict(testX)
    metrics = {}
    cm = confusion_matrix(testY, predictions).ravel()
    tn, fp, fn, tp = cm
    metrics['accuracy'] = (tp + tn) / float(cm.sum())
    metrics['sensitivity'] = tp / float(tp + fn)
    metrics['specificity'] = tn / float(tn + fp)
    print(f"[INFO] Metrics: {metrics}")
    return metrics

#Output screen classifying healthy or parkinson based on the image detection
def visualize_predictions(model, testingPaths, label_encoder):
    idxs = np.arange(0, len(testingPaths))
    idxs = np.random.choice(idxs, size=(25,), replace=False)
    images = []

    for i in idxs:
        image = cv2.imread(testingPaths[i])
        output = image.copy()
        output = cv2.resize(output, (128, 128))
        image = preprocess_image(image)
        features = extract_features(image)
        preds = model.predict([features])
        label = label_encoder.inverse_transform(preds)[0]
        color = (0, 255, 0) if label == "healthy" else (0, 0, 255)
        cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        images.append(output)

    montage = build_montages(images, (128, 128), (5, 5))[0]
    cv2.imshow("Output", montage)
    cv2.waitKey(0)

#main Function
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True, help="Path to input dataset")
    args = vars(ap.parse_args())

    # Paths for spiral and wave categories
    trainingPaths = [os.path.join(args["dataset"], "drawings", cat, "training") for cat in ["spiral", "wave"]]
    testingPaths = [os.path.join(args["dataset"], "drawings", cat, "testing") for cat in ["spiral", "wave"]]

    # Load data
    trainX, trainY = [], []
    testX, testY = [], []

    for path in trainingPaths:
        data, labels = load_data(path)
        trainX.extend(data)
        trainY.extend(labels)

    for path in testingPaths:
        data, labels = load_data(path)
        testX.extend(data)
        testY.extend(labels)

    trainX, trainY = np.array(trainX), np.array(trainY)
    testX, testY = np.array(testX), np.array(testY)

    print(f"[INFO] Loaded {len(trainX)} training samples and {len(testX)} testing samples.")

    # Encode labels
    le = LabelEncoder()
    trainY = le.fit_transform(trainY)
    testY = le.transform(testY)

    # Define models to evaluate
    # models = {
    #     "Random Forest": RandomForestClassifier(n_estimators=100),
    #     "Decision Tree": DecisionTreeClassifier(),
    #     "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    #     "Support Vector Machine": SVC(kernel="linear", probability=True),
    #     "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    # }

    # # Train the model
    # print("[INFO] Training the model...")
    # model = RandomForestClassifier(n_estimators=100)
    # model.fit(trainX, trainY)


''' 
.
.
Training model and evaluation the accuracy 
.
.
'''

# Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    model.fit(trainX, trainY)

# Evaluate
    predictions = model.predict(testX)
    print("Random Forest Classification Report:")
    print(classification_report(testY, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(testY, predictions))

 # Evaluate the model
    evaluate_model(model, testX, testY, le)

# Train Decision Tree
    model = DecisionTreeClassifier()
    model.fit(trainX, trainY)

# Evaluate
    predictions = model.predict(testX)
    print("Decision Tree Classification Report:")
    print(classification_report(testY, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(testY, predictions))

 # Evaluate the model
    evaluate_model(model, testX, testY, le)

# Train KNN
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(trainX, trainY)

# Evaluate
    predictions = model.predict(testX)
    print("K-Nearest Neighbors Classification Report:")
    print(classification_report(testY, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(testY, predictions))

 # Evaluate the model
    evaluate_model(model, testX, testY, le)

# Train SVM
    model = SVC(kernel="linear", probability=True)  # Linear kernel
    model.fit(trainX, trainY)

# Evaluate
    predictions = model.predict(testX)
    print("Support Vector Machine Classification Report:")
    print(classification_report(testY, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(testY, predictions))

 # Evaluate the model
    evaluate_model(model, testX, testY, le)

# Train XGBoost
#     model = XGBClassifier(
#     n_estimators=200,       # Number of trees in the ensemble
#     max_depth=6,            # Maximum depth of each tree
#     learning_rate=0.1,      # Step size shrinkage
#     subsample=0.8,          # Fraction of samples used per tree
#     colsample_bytree=0.8,   # Fraction of features used per tree
#     use_label_encoder=False,
#     eval_metric="logloss"
# )
    model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)


    model.fit(trainX, trainY)

# Evaluate
    predictions = model.predict(testX)
    print("XGBoost Classification Report:")
    print(classification_report(testY, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(testY, predictions))

 # Evaluate the model
    evaluate_model(model, testX, testY, le)

    # # Train the model
    # print("[INFO] Training the model...")
    # model = DecisionTressClassifier(n_estimators=100)
    # model.fit(trainX, trainY)

    # # Train and evaluate each model
    # for model_name, model in models.items():
    #     evaluate_model(model_name, model, trainX, trainY, testX, testY, le)

    

    # Visualize predictions
    testingImages = []
    for path in testingPaths:
        testingImages.extend(list_images(path))
    visualize_predictions(model, testingImages, le)


if __name__ == "__main__":
    main()
