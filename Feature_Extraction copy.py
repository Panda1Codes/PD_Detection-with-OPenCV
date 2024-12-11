import cv2
import numpy as np
import os
from math import sqrt
from scipy.signal import find_peaks
import csv

# Load an image and preprocess it
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)  # Binary inversion
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Feature extraction functions
def calculate_velocity(points, timestamps):
    velocities = []
    horz_velocities = []
    vert_velocities = []
    magnitude = []
    for i in range(1, len(points)):
        dx = points[i][0] - points[i - 1][0]
        dy = points[i][1] - points[i - 1][1]
        dt = timestamps[i] - timestamps[i - 1] if timestamps[i] != timestamps[i - 1] else 1
        velocity = (dx / dt, dy / dt)
        horz_velocities.append(velocity[0])
        vert_velocities.append(velocity[1])
        velocities.append(velocity)
        magnitude.append(sqrt(velocity[0]**2 + velocity[1]**2))
    return velocities, horz_velocities, vert_velocities, magnitude

def calculate_surface_time(points, pressure, timestamps):
    in_air_time = 0
    on_surface_time = 0
    for i in range(1, len(points)):
        dt = timestamps[i] - timestamps[i - 1]
        if pressure[i] > 0:
            on_surface_time += dt
        else:
            in_air_time += dt
    total_time = in_air_time + on_surface_time
    normalized_in_air = in_air_time / total_time if total_time > 0 else 0
    normalized_on_surface = on_surface_time / total_time if total_time > 0 else 0
    in_air_surface_ratio = in_air_time / on_surface_time if on_surface_time > 0 else 0
    return in_air_time, on_surface_time, normalized_in_air, normalized_on_surface, in_air_surface_ratio


def calculate_acceleration(velocities, timestamps):
    accelerations = []
    for i in range(1, len(velocities)):
        dt = timestamps[i] - timestamps[i - 1] if timestamps[i] != timestamps[i - 1] else 1
        ax = (velocities[i][0] - velocities[i - 1][0]) / dt
        ay = (velocities[i][1] - velocities[i - 1][1]) / dt
        accelerations.append((ax, ay))
    return accelerations

def count_direction_changes(values):
    changes = 0
    for i in range(1, len(values)):
        if values[i] * values[i - 1] < 0:  # Change in sign
            changes += 1
    return changes

def extract_features_from_stroke(points, timestamps, pressure):
    # Velocity
    velocities, horz_vel, vert_vel, vel_mag = calculate_velocity(points, timestamps)

    # Acceleration
    accelerations = calculate_acceleration(velocities, timestamps)
    horz_acc = [a[0] for a in accelerations]
    vert_acc = [a[1] for a in accelerations]
    acc_mag = [sqrt(a[0]**2 + a[1]**2) for a in accelerations]

    # Direction Changes
    ncv = count_direction_changes(horz_vel) + count_direction_changes(vert_vel)
    nca = count_direction_changes(horz_acc) + count_direction_changes(vert_acc)

    # Surface Time
    in_air_time, on_surface_time, norm_in_air, norm_on_surface, in_air_surface_ratio = calculate_surface_time(points, pressure, timestamps)

    # Combine Features
    features = {
        "Mean_Vel": np.mean(vel_mag),
        "Mean_Acc": np.mean(acc_mag),
        "NCV": ncv,
        "NCA": nca,
        "In_Air_Time": in_air_time,
        "On_Surface_Time": on_surface_time,
        "Norm_In_Air": norm_in_air,
        "Norm_On_Surface": norm_on_surface,
        "In_Air/On_Surface_Ratio": in_air_surface_ratio,
    }
    return features
    

# Process each folder
def process_dataset(dataset_path, output_csv):
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["File", "Type", "Condition", "Total_Strokes", "Mean_Vel", 
                         "In_Air_Time", "On_Surface_Time", "Norm_In_Air", 
                         "Norm_On_Surface", "In_Air/On_Surface_Ratio"])
        
        for root, dirs, files in os.walk(dataset_path):  # Recursively traverse the dataset
            for filename in files:
                if filename.endswith(".png"):  # Process only images
                    file_path = os.path.join(root, filename)
                    # Extract type (spiral/wave) and condition (healthy/parkinson) from folder names
                    parts = root.split(os.sep)
                    drawing_type = parts[-3] if len(parts) > 2 else "Unknown"
                    condition = parts[-1] if len(parts) > 0 else "Unknown"
                    
                    # Load image and preprocess
                    timestamps = np.arange(0, 100, 1)  # Dummy timestamps (replace with real values)
                    pressure = np.random.randint(0, 2, len(timestamps))  # Dummy pressure (replace with real values)
                    contours = load_and_preprocess_image(file_path)
                    
                    total_strokes = len(contours)
                    for contour in contours:
                        points = contour.squeeze()
                        if len(points) < 2:
                            continue
                        _, _, _, vel_magnitude = calculate_velocity(points, timestamps)
                        in_air_time, on_surface_time, norm_in_air, norm_on_surface, in_air_surface_ratio = calculate_surface_time(points, pressure, timestamps)
                        
                        # Write features to CSV
                        writer.writerow([file_path, drawing_type, condition, total_strokes, 
                                         np.mean(vel_magnitude), in_air_time, on_surface_time, 
                                         norm_in_air, norm_on_surface, in_air_surface_ratio])

# Run the script
dataset_path = "PD_Detection-with-OpenCV/Dataset/drawings"
output_csv = "features_extracted.csv"
process_dataset(dataset_path, output_csv)
