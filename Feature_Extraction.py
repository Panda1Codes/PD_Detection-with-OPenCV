import cv2
import numpy as np
from math import sqrt
from scipy.signal import find_peaks
import csv
import os

# Load an image and preprocess it
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)  # Binary inversion
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Calculate velocity
def calculate_velocity(points, timestamps):
    velocities = []
    horz_velocities = []
    vert_velocities = []
    magnitude = []
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        dt = timestamps[i] - timestamps[i - 1] if timestamps[i] != timestamps[i - 1] else 1  # Avoid division by 0
        velocity = (dx / dt, dy / dt)
        horz_velocities.append(velocity[0])
        vert_velocities.append(velocity[1])
        velocities.append(velocity)
        magnitude.append(sqrt(velocity[0]**2 + velocity[1]**2))
    return velocities, horz_velocities, vert_velocities, magnitude

# Calculate acceleration
def calculate_acceleration(velocities, timestamps):
    accelerations = []
    horz_accelerations = []
    vert_accelerations = []
    magnitude = []
    for i in range(1, len(velocities)):
        dvx = velocities[i][0] - velocities[i - 1][0]
        dvy = velocities[i][1] - velocities[i - 1][1]
        dt = timestamps[i] - timestamps[i - 1] if timestamps[i] != timestamps[i - 1] else 1  # Avoid division by 0
        acceleration = (dvx / dt, dvy / dt)
        horz_accelerations.append(acceleration[0])
        vert_accelerations.append(acceleration[1])
        accelerations.append(acceleration)
        magnitude.append(sqrt(acceleration[0]**2 + acceleration[1]**2))
    return accelerations, horz_accelerations, vert_accelerations, magnitude

# Calculate jerk
def calculate_jerk(accelerations, timestamps):
    jerks = []
    horz_jerks = []
    vert_jerks = []
    magnitude = []
    for i in range(1, len(accelerations)):
        dax = accelerations[i][0] - accelerations[i - 1][0]
        day = accelerations[i][1] - accelerations[i - 1][1]
        dt = timestamps[i] - timestamps[i - 1] if timestamps[i] != timestamps[i - 1] else 1  # Avoid division by 0
        jerk = (dax / dt, day / dt)
        horz_jerks.append(jerk[0])
        vert_jerks.append(jerk[1])
        jerks.append(jerk)
        magnitude.append(sqrt(jerk[0]**2 + jerk[1]**2))
    return jerks, horz_jerks, vert_jerks, magnitude

# Count changes in velocity direction
def count_direction_changes(values):
    changes = 0
    for i in range(1, len(values)):
        if (values[i] > 0 and values[i - 1] < 0) or (values[i] < 0 and values[i - 1] > 0):
            changes += 1
    return changes

# Calculate NCV and NCA
def calculate_relative_changes(count_changes, total_length):
    if total_length == 0:
        return 0
    return count_changes / total_length

# Process in-air and on-surface time
def calculate_surface_time(points, pressure, timestamps):
    in_air_time = 0
    on_surface_time = 0
    for i in range(1, len(points)):
        dt = timestamps[i] - timestamps[i - 1]
        if pressure[i] > 0:  # Assuming pressure > 0 means "on surface"
            on_surface_time += dt
        else:
            in_air_time += dt
    total_time = in_air_time + on_surface_time
    normalized_in_air = in_air_time / total_time if total_time > 0 else 0
    normalized_on_surface = on_surface_time / total_time if total_time > 0 else 0
    in_air_surface_ratio = in_air_time / on_surface_time if on_surface_time > 0 else 0
    return in_air_time, on_surface_time, normalized_in_air, normalized_on_surface, in_air_surface_ratio

# Extract features from contours
def extract_features_from_contours(contours, timestamps, pressure):
    total_strokes = len(contours)
    all_features = []
    for contour in contours:
        points = contour.squeeze()  # Flatten contour points
        if len(points) < 2:  # Skip small contours
            continue

        # Calculate features
        velocities, horz_velocities, vert_velocities, vel_magnitude = calculate_velocity(points, timestamps)
        accelerations, horz_accelerations, vert_accelerations, acc_magnitude = calculate_acceleration(velocities, timestamps)
        jerks, horz_jerks, vert_jerks, jerk_magnitude = calculate_jerk(accelerations, timestamps)

        num_vel_changes = count_direction_changes(velocities)
        num_acc_changes = count_direction_changes(accelerations)

        relative_ncv = calculate_relative_changes(num_vel_changes, len(points))
        relative_nca = calculate_relative_changes(num_acc_changes, len(points))

        in_air_time, on_surface_time, norm_in_air, norm_on_surface, in_air_surface_ratio = calculate_surface_time(points, pressure, timestamps)

        features = [
            total_strokes,
            np.mean(vel_magnitude),
            np.mean(acc_magnitude),
            np.mean(jerk_magnitude),
            num_vel_changes,
            num_acc_changes,
            relative_ncv,
            relative_nca,
            in_air_time,
            on_surface_time,
            norm_in_air,
            norm_on_surface,
            in_air_surface_ratio
        ]
        all_features.append(features)
    return all_features

# Main function
def process_dataset(dataset_path, output_csv):
    # Convert to absolute path
    dataset_path = os.path.abspath(dataset_path)
    print("Resolved dataset path:", dataset_path)
    
    # Check if directory exists
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"The dataset path does not exist: {dataset_path}")
    
    # List files
    files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.png')]
    if not files:
        raise FileNotFoundError(f"No .png files found in the dataset path: {dataset_path}")
    
    # Process each file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["File", "Total_Strokes", "Mean_Vel", "Mean_Acc", "Mean_Jerk",
                         "Num_Vel_Changes", "Num_Acc_Changes", "Relative_NCV", "Relative_NCA",
                         "In_Air_Time", "On_Surface_Time", "Norm_In_Air", "Norm_On_Surface", "In_Air/On_Surface_Ratio"])
        for file_path in files:
            print(f"Processing file: {file_path}")
            timestamps = np.arange(0, 100, 1)  # Dummy timestamps (replace with actual data)
            pressure = np.random.randint(0, 2, len(timestamps))  # Dummy pressure (replace with actual data)
            contours = load_and_preprocess_image(file_path)
            features = extract_features_from_contours(contours, timestamps, pressure)
            for feature_set in features:
                writer.writerow([file_path] + feature_set)

# Run the script
dataset_path = os.path.join('Dataset', 'drawings')
print("Resolved dataset path:", dataset_path)

output_csv = "features_extracted.csv"
process_dataset(dataset_path, output_csv)
