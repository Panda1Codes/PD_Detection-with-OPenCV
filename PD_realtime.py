import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from skimage import feature
from skimage.feature import local_binary_pattern
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.utils import shuffle 
from sklearn.model_selection import StratifiedKFold

def detect_paper(image):
    """
    Detects a rectangular paper-like object in the image.

    Args:
        image (numpy.ndarray): Input image.

    Returns:
        list: Coordinates of the four corners of the detected paper, or None if not found.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours to find a quadrilateral
    for contour in contours:
        # Filter small contours
        if cv2.contourArea(contour) < 1000:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # Check if the contour has four vertices and is convex
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2)

    return None


def apply_perspective_transform(image, corners):
    """
    Applies a perspective transform to extract the paper region from the image
    based on the detected paper's corners.

    Args:
        image (numpy.ndarray): Input image.
        corners (list): Coordinates of the four corners of the paper.

    Returns:
        numpy.ndarray: Warped image with the paper region, or None if failed.
    """
    # Order the corners to consistently define the paper
    corners = sorted(corners, key=lambda x: (x[1], x[0]))
    top_left, top_right = sorted(corners[:2], key=lambda x: x[0])
    bottom_left, bottom_right = sorted(corners[2:], key=lambda x: x[0])
    ordered_corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

    # Compute width and height of the paper
    width_top = np.linalg.norm(top_right - top_left)
    width_bottom = np.linalg.norm(bottom_right - bottom_left)
    height_left = np.linalg.norm(top_left - bottom_left)
    height_right = np.linalg.norm(top_right - bottom_right)

    max_width = int(max(width_top, width_bottom))
    max_height = int(max(height_left, height_right))

    # Destination points for the perspective transform
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype='float32')

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(ordered_corners, dst)

    # Apply the perspective warp
    try:
        warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
        return warped
    except cv2.error as e:
        print(f"[ERROR] Perspective transform failed: {e}")
        return None


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

    # Combine ORB and Harris keypoints
    combined_keypoints1 = list(keypoints1) + harris_keypoints1 if keypoints1 else harris_keypoints1
    combined_keypoints2 = list(keypoints2) + harris_keypoints2 if keypoints2 else harris_keypoints2

    # Compute descriptors for the combined keypoints using ORB
    combined_descriptors1 = orb.compute(img1, combined_keypoints1)[1] if combined_keypoints1 else None
    combined_descriptors2 = orb.compute(img2, combined_keypoints2)[1] if combined_keypoints2 else None

    if combined_descriptors1 is None or combined_descriptors2 is None:
        print("[WARNING] One or both descriptor arrays are empty.")
        return [], []

    if combined_descriptors1.shape[1] != combined_descriptors2.shape[1]:
        print("[WARNING] Descriptor dimension mismatch.")
        return [], []

    # Match features using brute-force matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(combined_descriptors1, combined_descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)  # Sort by distance (similarity)

    keypoints_ref = [combined_keypoints1[m.queryIdx].pt for m in matches]
    keypoints_edge = [combined_keypoints2[m.trainIdx].pt for m in matches]

    return keypoints_ref, keypoints_edge


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
    if len(image.shape) > 2:  # Check if the image is not already grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, resize_dim)
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return image


def extract_features(image, P=8, R=1):
    # Compute HOG features
    # Initialize the HOG Descriptor
    hog = cv2.HOGDescriptor(_winSize=(image.shape[1] // 10 * 10, image.shape[0] // 10 * 10),
                            _blockSize=(20, 20),  # cells_per_block (2x2) * pixels_per_cell (10x10)
                            _blockStride=(10, 10),  # same as pixels_per_cell
                            _cellSize=(10, 10),
                            _nbins=9)

    # Compute HOG features
    hog_features = hog.compute(image)
    hog_features = hog_features.flatten()  # Flatten the result into a 1D array


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
    model = RandomForestClassifier(n_estimators=51, max_depth=6, min_samples_split=6, min_samples_leaf=2, random_state=42, max_features="log2")
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
        data, labels = load_testing_data(path)
        testX.extend(data)
        testY.extend(labels)

    testX = np.array(testX)
    testY = np.array(testY)

    # Compute testing accuracy
    predictions = ensemble.predict(testX)
    testing_accuracy = np.mean(predictions == testY) * 100
    print(f"Testing Accuracy: {testing_accuracy:.2f}%")

    print("[INFO] Starting real-time detection...")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Unable to capture frame. Exiting...")
            break

        # Process frame with Gamma Correction and Gaussian Blur
        processed_frame = process_frame_for_detection(frame, gamma=1.2, blur_ksize=(5, 5))

        # Detect paper corners
        paper_corners = detect_paper(processed_frame)

        if paper_corners is not None:
            # Draw the detected paper region on the frame
            cv2.polylines(frame, [np.int32(paper_corners)], isClosed=True, color=(0, 255, 0), thickness=2)
            warped_frame = apply_perspective_transform(frame, paper_corners)
            if warped_frame is not None:
                # Show the warped paper region
                cv2.imshow("Warped Paper", warped_frame)

                # Classify the detected paper region
                detected_label = classify_image(ensemble, label_encoder, warped_frame)
                print(f"[INFO] Detected: {detected_label}")

                # Compare with reference
                reference_path = healthy_reference if detected_label == "healthy" else unhealthy_reference
                title = "Healthy Reference" if detected_label == "healthy" else "Unhealthy Reference"
                compare_with_reference(warped_frame, reference_path, title, training_accuracy, testing_accuracy)

        else:
            cv2.putText(frame, "No paper detected", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Display real-time feed
        cv2.putText(frame, "Press 's' to classify, 'q' to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Real-Time Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()


