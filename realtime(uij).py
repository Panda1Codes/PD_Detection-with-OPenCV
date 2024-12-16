import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from skimage import feature

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
    based on the detected paper's actual size.

    Args:
        image (numpy.ndarray): Input image.
        corners (list): Coordinates of the four corners of the paper.

    Returns:
        numpy.ndarray: Warped image with the paper region.
    """
    # Order corners to consistently define the paper
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
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return warped

def expand_bounding_box(bbox, shape, scale = 1.5):
    x, y, w, h = bbox
    W = shape[0]
    H = shape[1]
    # Calculate center coordinate
    cx, cy = x + w // 2, y + h // 2
    
    # Calculate new width, height
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Calculate new coordinate (Expand size from center coordinate)
    new_x = max(0,int(cx - new_w // 2))
    new_y = max(0,int(cy - new_h // 2))

    # Clipping
    new_w = min(new_w, W - new_x)  
    new_h = min(new_h, H - new_y)
    
    return (new_x, new_y, new_w, new_h)


def preprocess_image(image, resize_dim=(200, 200)):
    # Convert to grayscale, resize, and apply thresholding
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, resize_dim)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
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

    # Create CSRT Tracker, Initialize tracking variable
    tracker = cv2.legacy.TrackerCSRT_create()
    tracking_initialized = False

    while True:
        ret, frame = cap.read()
        H = frame.shape[0]
        W = frame.shape[1]
        if not ret:
            print("[ERROR] Unable to capture frame. Exiting...")
            break

        # Display live camera feed
        cv2.putText(frame, "Show a spiral or wave drawing to classify", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if not tracking_initialized:
            # Detect the paper in real-time
            corners = detect_paper(frame)
            if corners is not None:
                # Initialize the tracker with detected region
                bbox = cv2.boundingRect(corners)
                tracker.init(frame, bbox)
                tracking_initialized = True

        else:
            # Update tracker and draw the tracked region
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = expand_bounding_box(list(map(int, bbox)),(W,H),1.5)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # Expanded x,y-coordinate must be within frame
                if (y>=0)&(x>=0)&(y+h<H)&(x+h<W):
                    # Detect paper in tracked region
                    corners = detect_paper(frame[y:y + h, x:x + w])
                    if corners is not None:
                        corners = sorted(corners, key=lambda x: (x[1], x[0]))
                        top_left, top_right = sorted(corners[:2], key=lambda x: x[0])
                        bottom_left, bottom_right = sorted(corners[2:], key=lambda x: x[0])
                        bias = lambda f : [f[0]+x,f[1]+y] # Bias for cropped frame to full frame
                        corners = np.array([bias(top_left), bias(top_right), bias(bottom_right), bias(bottom_left)], dtype='int32')
            else:
                # Initialize variables if tracking fails
                tracking_initialized = False
        
        if corners is not None:
            # Draw contours on the original frame for visualization
            cv2.polylines(frame, [corners.astype(int)], isClosed=True, color=(0, 255, 0), thickness=2)
        else:
            cv2.putText(frame, "Not detected", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Press 's' to capture and classify the current frame
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            # Preprocess and classify the image
            if corners is not None:
                warped_frame = apply_perspective_transform(frame, corners)
                detected_label = classify_image(model, label_encoder, warped_frame)
                cv2.imshow("preprocess warped frame", preprocess_image(warped_frame))
                cv2.putText(warped_frame, f"Detected: {detected_label}",(10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imshow("warped frame", warped_frame)

        # Press 'r' to reset tracking boundary box
        if key==ord("r"):
            tracking_initialized = False

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
