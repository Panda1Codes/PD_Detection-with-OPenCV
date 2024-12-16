import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
# from sklearn.neighbors import KNeighborsClassifier
# from xgboost import XGBClassifier
# from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from skimage import feature
from skimage.feature import local_binary_pattern
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.utils import shuffle 
from sklearn.model_selection import StratifiedKFold

def apply_gamma_correction(image, gamma=1.0):
    """
    Apply gamma correction to an image.
    """
    gamma_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, gamma_table)

def apply_gaussian_blur(image, blur_ksize=(5, 5)):
    """
    Apply Gaussian blur to an image.
    """
    return cv2.GaussianBlur(image, blur_ksize, 0)

def preprocess_image(image, resize_dim=(200, 200)):
    """
    Preprocesses an image: converts to grayscale, resizes, and thresholds.
    """
    # image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    if len(image.shape) > 2:  # Check if the image is not already grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, resize_dim)
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return image


def extract_features(image, P=8, R=1):
    # Compute HOG features
    hog_features = feature.hog(image, orientations=9, 
                                pixels_per_cell=(10, 10), 
                                cells_per_block=(2, 2), 
                                transform_sqrt=True, block_norm="L1")

    # Find contours to calculate additional features
    contours, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_strokes = len(contours)
    total_stroke_length = sum(cv2.arcLength(contour, closed=True) for contour in contours)

    # Compute Local Binary Pattern (LBP) features
    lbp_features = feature.local_binary_pattern(image, P, R, method="uniform")
    hist, _ = np.histogram(lbp_features.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  # Normalize histogram

    # Flatten and combine all features into a single vector
    contour_features = [num_strokes, total_stroke_length]  # Contour features as a list
    feature_vector = np.hstack([hog_features, contour_features, hist])  # Concatenate into a single array

    return feature_vector

#Data Augmentation, since the dataset is quite small..
def augment_image(image):
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 15, 1)
    rotated = cv2.warpAffine(image, M, (cols, rows))
    return rotated

#Training the model
# def train_model(trainingPaths):
#     data, labels = [], []
#     for path in trainingPaths:
#         imagePaths = [os.path.join(path, subdir, img) 
#                       for subdir in ["healthy", "parkinson"]
#                       for img in os.listdir(os.path.join(path, subdir))]
        
#         for imgPath in imagePaths:
#             label = os.path.basename(os.path.dirname(imgPath))
#             image = cv2.imread(imgPath)
#             if image is None:
#                 print(f"[WARNING] Could not read image {imgPath}. Skipping.")
#                 continue
#             preprocessed = preprocess_image(image)
#             augmented = augment_image(preprocessed)

#             features = extract_features(preprocessed)
#             augmented_features = extract_features(augmented)

#             data.append(features)
#             labels.append(label)
#             data.append(augmented_features)
#             labels.append(label)

#     data, labels = shuffle(data, labels, random_state=42)
#     le = LabelEncoder()
#     labels = le.fit_transform(labels)

#     #hyperparameter tuning for enhance and resize image to get more better features:
#     param_grid = {
#         'n_estimators': [100, 200],
#         'max_depth': [3, 5],
#         'learning_rate': [0.01, 0.1],
#         'subsample': [0.8, 1.0]
#     }

#     # model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
#     # model.fit(data, labels)

#     # model = LGBMClassifier(n_estimators=100, max_depth=-1, random_state=42)
#     # model.fit(data, labels)
    
#     # # model = KNeighborsClassifier(n_neighbors=3)
#     # # model.fit(data, labels)   

#     # # model = LogisticRegression(max_iter=1000, random_state=42)
#     # # model.fit(data,labels)

#     # model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
#     # model.fit(data,labels)

#     xgb = XGBClassifier(eval_metric="logloss", random_state=42)
#     grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring ='accuracy', cv=3, verbose=1)
#     grid_search.fit(data, labels)

#     print(f"[INFO] Best Parameters: {grid_search.best_params_}")
#     model = grid_search.best_estimator_

#     #Ensemeble Model
#     rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
#     rf_model.fit(data, labels)

#     ensemble = StackingClassifier(estimators=[
#         ('rf', rf_model),
#         ('xgb', model)
#     ], final_estimator=RandomForestClassifier(n_estimators=100, random_state=42))
#     ensemble.fit(data, labels)

#     predictions = ensemble.predict(data)
#     accuracy = np.mean(predictions == labels) * 100
#     print(f"Training Accuracy: {accuracy:.2f}%")
#     return ensemble, le, accuracy


def train_model(trainingPaths):
    data, labels = [], []
    for path in trainingPaths:
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
            augmented = augment_image(preprocessed)

            features = extract_features(preprocessed)
            augmented_features = extract_features(augmented)

            data.append(features)
            labels.append(label)
            data.append(augmented_features)
            labels.append(label)

    data, labels = shuffle(data, labels, random_state=42)
    le = LabelEncoder()
    labels = le.fit_transform(labels)

    # Define and train Random Forest model
    print("[INFO] Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_split=4, random_state=42)
    model.fit(data, labels)

    # Evaluate Training Accuracy
    predictions = model.predict(data)
    training_accuracy = np.mean(predictions == labels) * 100
    print(f"Training Accuracy: {training_accuracy:.2f}%")
    
    return model, le, training_accuracy


def detect_spiral_or_wave(image):
    """
    Detect if the captured shape resembles a spiral or wave using contour properties.
    """
    contours, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        circularity = 4 * np.pi * cv2.contourArea(contour) / (cv2.arcLength(contour, True) ** 2)
        if circularity > 0.7:
            return "spiral"
        else:
            return "wave"
    return "unknown"

def classify_and_detect(model, label_encoder, frame):
    preprocessed = preprocess_image(frame)
    detected_shape = detect_spiral_or_wave(preprocessed)
    if detected_shape not in ["spiral", "wave"]:
        return "unknown", None

    features = extract_features(preprocessed)
    prediction = model.predict([features])[0]
    label = label_encoder.inverse_transform([prediction])[0]
    return label, detected_shape


def classify_image(ensemble, label_encoder, image):
    preprocessed = preprocess_image(image)
    features = extract_features(preprocessed)
    prediction = ensemble.predict([features])[0]
    label = label_encoder.inverse_transform([prediction])[0]
    return label


def compare_with_reference(image, reference_path, title, training_accuracy, testing_accuracy):
    if not os.path.exists(reference_path):
        print(f"[ERROR] Reference image not found: {reference_path}")
        return

    reference_image = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    if reference_image is None:
        print(f"[ERROR] Unable to load reference image: {reference_path}")
        return

    # Preprocess the reference image
    reference_image = preprocess_image(reference_image)

    # Perform edge detection on the captured frame
    edge_image = cv2.Canny(image, 100, 200)

    # Resize images to ensure the same dimensions
    h, w = edge_image.shape
    reference_image = cv2.resize(reference_image, (w, h))

    # Concatenate images horizontally
    combined = cv2.hconcat([reference_image, edge_image])

    # Find and draw connecting lines for similar features
    keypoints_ref, keypoints_edge = find_similar_features(reference_image, edge_image)
    combined = draw_feature_lines(combined, keypoints_ref, keypoints_edge, offset=w)

    # Overlay training and testing accuracy
    cv2.putText(combined, f"Training Acc: {training_accuracy:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined, f"Testing Acc: {testing_accuracy:.2f}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Display the result
    cv2.imshow(title, combined)
    cv2.waitKey(0)


def find_similar_features(img1, img2):
    """
    Find similar features between two images using ORB and Harris Corner Detection.
    """
    # ORB feature detection
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # Harris Corner Detection
    def detect_harris_corners(image):
        corners = cv2.cornerHarris(image, blockSize=2, ksize=3, k=0.04)
        corners = cv2.dilate(corners, None)  # Dilate to enhance corner points
        threshold = 0.01 * corners.max()
        keypoints = [cv2.KeyPoint(float(x), float(y), 1) for y, x in zip(*np.where(corners > threshold))]
        return keypoints

    harris_keypoints1 = detect_harris_corners(img1)
    harris_keypoints2 = detect_harris_corners(img2)

    # Convert ORB keypoints to list and combine with Harris keypoints
    combined_keypoints1 = list(keypoints1) + harris_keypoints1
    combined_keypoints2 = list(keypoints2) + harris_keypoints2

    # Extract descriptors for the combined keypoints using ORB
    combined_descriptors1 = orb.compute(img1, combined_keypoints1)[1]
    combined_descriptors2 = orb.compute(img2, combined_keypoints2)[1]

    # Match features using brute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(combined_descriptors1, combined_descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)  # Sort by distance (similarity)

    keypoints_ref = [combined_keypoints1[m.queryIdx].pt for m in matches]
    keypoints_edge = [combined_keypoints2[m.trainIdx].pt for m in matches]

    return keypoints_ref, keypoints_edge



def draw_feature_lines(image, keypoints1, keypoints2, offset=0):
    """
    Draw lines connecting matched features between two images.
    """
    for (x1, y1), (x2, y2) in zip(keypoints1, keypoints2):
        x2_offset = x2 + offset  # Adjust for concatenated image
        cv2.line(image, (int(x1), int(y1)), (int(x2_offset), int(y2)), (0, 255, 0), 1)
    return image


def load_testing_data(test_path):
    data, labels = [], []
    for subdir in ["healthy", "parkinson"]:
        subdir_path = os.path.join(test_path, subdir)
        if not os.path.exists(subdir_path):
            print(f"[WARNING] Subdirectory {subdir_path} does not exist. Skipping...")
            continue
        for imgPath in os.listdir(subdir_path):
            full_path = os.path.join(subdir_path, imgPath)
            image = cv2.imread(full_path)
            if image is None:
                print(f"[WARNING] Could not read image {full_path}. Skipping...")
                continue
            preprocessed = preprocess_image(image)
            features = extract_features(preprocessed)
            data.append(features)
            labels.append(subdir)

    le = LabelEncoder()
    labels = le.fit_transform(labels)
    return np.array(data), labels


def process_frame_for_detection(frame, gamma=1.0, blur_ksize=(5, 5)):
    """
    Apply gamma correction and Gaussian blur to a video frame.
    """
    # Apply Gamma Correction
    frame = apply_gamma_correction(frame, gamma)

    # Apply Gaussian Blur
    frame = apply_gaussian_blur(frame, blur_ksize)

    return frame


def main():
    dataset_path = input("[INPUT] Enter dataset path for training: ").strip()
    healthy_reference = r"C:\Users\priya\OneDrive\Desktop\Notes\Media Programming\Final Project_PD Detection\dataset\drawings\spiral\testing\healthy\V02HE01.png"
    unhealthy_reference = r"C:\Users\priya\OneDrive\Desktop\Notes\Media Programming\Final Project_PD Detection\dataset\drawings\spiral\testing\parkinson\V07PE01.png"

    # Paths for spiral and wave categories
    trainingPaths = [os.path.join(dataset_path, "drawings", cat, "training") for cat in ["spiral", "wave"]]
    testingPaths = [os.path.join(dataset_path, "drawings", cat, "testing") for cat in ["spiral", "wave"]]

    # Train the model
    ensemble, label_encoder, training_accuracy = train_model(trainingPaths)

    # Load testing data
    testX, testY = [], []
    for path in testingPaths:
        data, labels = load_testing_data(path)  # Define a function to load testing data
        testX.extend(data)
        testY.extend(labels)

    testX = np.array(testX)
    testY = np.array(testY)

    # # Compute testing accuracy
    # predictions = ensemble.predict(testX)
    # testing_accuracy = np.mean(predictions == testY) * 100
    # print(f"Testing Accuracy: {testing_accuracy:.2f}%")

    # Combine train (data, labels) and test (testX, testY) for overall accuracy
    combined_X = np.vstack([data, testX])  # data is the training features
    combined_Y = np.hstack([labels, testY])  # labels are the training labels

    # Predict combined accuracy
    combined_predictions = ensemble.predict(combined_X)
    overall_accuracy = np.mean(combined_predictions == combined_Y) * 100
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")

    scores = cross_val_score(ensemble, data, labels, cv=5, scoring="accuracy")
    print(f"Cross-Validation Accuracy: {np.mean(scores) * 100:.2f}%")



    print("[INFO] Starting real-time detection...")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Unable to capture frame. Exiting...") 
            break

        # Process frame with Gamma Correction and Gaussian Blur
        processed_frame = process_frame_for_detection(frame, gamma=1.2, blur_ksize=(5, 5))


#display the live feed with instructions
        frame_copy = frame.copy()
        cv2.putText(frame_copy, "Press 's' to classify or 'q' to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Real-Time Detection", frame_copy)

        key = cv2.waitKey(1) & 0xFF


        if key == ord("s"):
            detected_label = classify_image(ensemble, label_encoder, frame)
            print(f"[INFO] Detected: {detected_label}")

            reference_path = healthy_reference if detected_label == "healthy" else unhealthy_reference
            title = "Healthy Reference" if detected_label == "healthy" else "Unhealthy Reference"

            compare_with_reference(frame, reference_path, title, training_accuracy, overall_accuracy)

        if key == ord("q"):
            print("[INFO] Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()


