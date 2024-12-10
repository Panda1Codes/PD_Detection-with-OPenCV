import cv2
import numpy as np
import os
from math import sqrt
import matplotlib.pyplot as plt
import csv
import time 

def timeit(fn):
    def wrapper(*args, **kwargs):
        start=time()
        res=fn(*args, **kwargs)
        print(fn.__name__, "took", time()-start, "seconds.")
        return res
    return wrapper

global control_data_path
control_data_path = os.path.join('Dataset','hw_dataset', 'control')
parkinson_data_path = os.path.join('Dataset','hw_dataset', 'parkinson')

output_csv_path = "pd_features.csv"

control_file_list = [os.path.join(control_data_path, x) for x in os.listdir(control_data_path)]
parkinson_file_list = [os.path.join(parkinson_data_path, x) for x in os.listdir(parkinson_data_path)]

'''
Features
->No of strokes
->Stroke speed
->Velocity
->Acceleration
->Jerk
->Horizontal velocity/acceleration/jerk
->Vertical velocity/acceleration/jerk
->Number of changes in velocity direction
->Number of changes in acceleration direction
->Relative NCV
->Relative NCA
->in-air time
->on-surface time
->normalized in-air time
->normalized on-surface time
->in-air/on-surface ratio
'''

header_row = ["X", "Y", "Z", "Pressure", "GripAngle", "Timestamp", "Test_ID"]

#@timeit
def get_no_strokes(df):
    pressure_data=df['Pressure'].as_matrix()
    on_surface = (pressure_data>600).astype(int)
    return ((np.roll(on_surface, 1) - on_surface) != 0).astype(int).sum()

#@timeit
def get_speed(df):
    total_dist=0
    duration=df['Timestamp'].as_matrix()[-1]
    coords=df[['X', 'Y', 'Z']].as_matrix()
    for i in range(10, df.shape[0]):
        temp=np.linalg.norm(coords[i, :]-coords[i-10, :])
        total_dist+=temp
    speed=total_dist/duration
    return speed

#@timeit
def get_in_air_time(data):
    data=data['Pressure'].as_matrix()
    return (data<600).astype(int).sum()

#@timeit
def get_on_surface_time(data):
    data=data['Pressure'].as_matrix()
    return (data>600).astype(int).sum()

def load_csv(file_path):
    data = np.genfromtxt(file_path, delimiter=';', skip_header=1)
    return data

def count_strokes(file_path):
    data = load_csv(file_path)
    static_test_data = data[data[:, 6] == 0]  # Use only static test
    initial_timestamp = static_test_data[0, 5]
    static_test_data[:, 5] -= initial_timestamp

    plt.plot(static_test_data[:, 5], static_test_data[:, 3])  # Timestamp vs Pressure
    plt.xlabel("Timestamp")
    plt.ylabel("Pressure")
    plt.title("Pressure over Time")
    plt.show()

def find_velocity(file_path):
    data = load_csv(file_path)
    static_test_data = data[data[:, 6] == 1]  # Use only static test
    initial_timestamp = static_test_data[0, 5]
    static_test_data[:, 5] -= initial_timestamp

    #change in direction and its position

    data_path = file_path 
    Vel = []
    horz_Vel = []
    horz_vel_mag = []
    vert_vel_mag = []
    vert_Vel = []
    magnitude = []
    timestamp_diff =  []

    print('No of Coordinates:', len(static_test_data))
    t = 0
    for i in range(len(static_test_data) - 10):
        if t + 10 < len(static_test_data):
            x_diff = static_test_data[t + 10, 0] - static_test_data[t, 0]
            y_diff = static_test_data[t + 10, 1] - static_test_data[t, 1]
            time_diff = static_test_data[t + 10, 5] - static_test_data[t, 5]

            if time_diff == 0:  # Avoid division by zero
                continue

            vel_x = x_diff / time_diff
            vel_y = y_diff / time_diff

            Vel.append((vel_x, vel_y))
            horz_Vel.append(vel_x)
            vert_Vel.append(vel_y)
            # magnitude_horz_vel.append(abs(vel_x))
            # magnitude_vert_vel.append(abs(vel_y))
            magnitude.append(sqrt(vel_x**2 + vel_y**2))
            timestamp_diff.append(time_diff)
            t += 10
        else:
            break

    magnitude_vel = np.mean(magnitude)
    magnitude_horz_vel = np.mean(np.abs(horz_Vel))
    magnitude_vert_vel = np.mean(np.abs(vert_Vel))

    print(magnitude_vel, magnitude_horz_vel, magnitude_vert_vel)
    return Vel, magnitude, timestamp_diff, horz_Vel, vert_Vel, magnitude_vel, magnitude_horz_vel, magnitude_vert_vel

    # print("Computed Values:")
    # print(f"Magnitude Velocity: {np.mean(magnitude) if magnitude else 0}")
    # print(f"Horizontal Velocity Magnitude: {np.mean(horz_vel_mag) if horz_vel_mag else 0}")
    # print(f"Vertical Velocity Magnitude: {np.mean(vert_vel_mag) if vert_vel_mag else 0}")

    # return {
    #     "data_path": file_path,
    #     "Vel": Vel,
    #     "horz_Vel": horz_Vel,
    #     "horz_vel_mag": horz_vel_mag,
    #     "vert_vel_mag": vert_vel_mag,
    #     "vert_Vel": vert_Vel,
    #     "magnitude": magnitude,
    #     "timestamp_diff": timestamp_diff
    # }

def find_acceleration(file_path):

    #To find change in direction and its velocity
    Vel, magnitude, timestamp_diff, horz_Vel, vert_Vel, magnitude_vel, horz_vel_mag, vert_vel_mag = find_velocity(file_path)

    accl = []
    horz_Accl = []
    vert_Accl = []
    magnitude = []

    print('No of Coordinates:', len(Vel))
    for i in range(len(Vel) - 1):
        time_diff = timestamp_diff[i]

        accel_x = (Vel[i + 1][0] - Vel[i][0]) / time_diff
        accel_y = (Vel[i + 1][1] - Vel[i][1]) / time_diff

        accl.append((accel_x, accel_y))
        horz_Accl.append(accel_x)
        vert_Accl.append(accel_y)
        magnitude.append(sqrt(accel_x**2 + accel_y**2))

    magnitude_acc = np.mean(magnitude)
    magnitude_horz_acc = np.mean(np.abs(horz_Accl))
    magnitude_vert_acc = np.mean(np.abs(vert_Accl))

    print(magnitude_acc, magnitude_horz_acc, magnitude_vert_acc)
    return accl, magnitude, horz_Accl, vert_Accl, timestamp_diff, magnitude_acc, magnitude_horz_acc, magnitude_vert_acc

def find_jerk(file_path):
    accl, magnitude, horz_Accl, vert_Accl, timestamp_diff, magnitude_acc, magnitude_horz_acc, magnitude_vert_acc = find_acceleration(file_path)

    jerk = []
    horz_Jerk = []
    vert_Jerk = []
    magnitude = []

    print('No of Coordinates:', len(accl))
    for i in range(len(accl) - 1):
        accel_x_diff = (accl[i + 1][0] - accl[i][0]) / timestamp_diff[i]
        accel_y_diff = (accl[i + 1][1] - accl[i][1]) / timestamp_diff[i]

        jerk.append((accel_x_diff, accel_y_diff))
        horz_Jerk.append(accel_x_diff)
        vert_Jerk.append(accel_y_diff)
        magnitude.append(sqrt(accel_x_diff**2 + accel_y_diff**2))

    magnitude_jerk = np.mean(magnitude)
    magnitude_horz_jerk = np.mean(np.abs(horz_Jerk))
    magnitude_vert_jerk = np.mean(np.abs(vert_Jerk))

    print(magnitude_jerk, magnitude_horz_jerk, magnitude_vert_jerk)
    return jerk, magnitude, horz_Jerk, vert_Jerk, magnitude_jerk, magnitude_horz_jerk, magnitude_vert_jerk

def NCV_per_halfcircle(file_path):
    data = load_csv(file_path)
    static_test_data = data[data[:, 6] == 0]  # Use only static test
    initial_timestamp = static_test_data[0, 5]
    static_test_data[:, 5] -= initial_timestamp

    Vel = []
    ncv = []
    temp_ncv = 0
    basex = static_test_data[0, 0]
    for i in range(len(static_test_data) - 1):
        if static_test_data[i, 0] == basex:
            ncv.append(temp_ncv)
            temp_ncv = 0
            continue

        vel_x = (static_test_data[i + 1, 0] - static_test_data[i, 0]) / (static_test_data[i + 1, 5] - static_test_data[i, 5])
        vel_y = (static_test_data[i + 1, 1] - static_test_data[i, 1]) / (static_test_data[i + 1, 5] - static_test_data[i, 5])

        if (vel_x, vel_y) != (0, 0):
            temp_ncv += 1

    ncv.append(temp_ncv)
    ncv_val = np.sum(ncv) / np.count_nonzero(ncv)
    print(ncv_val)
    return ncv, ncv_val

# Function to save features to CSV
def save_features_to_csv(file_path, features):
    with open(output_csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        # Write header only if file is empty
        if file.tell() == 0:
            writer.writerow(["File", "Magnitude_Vel", "Hor_Vel", "Vert_Vel",
                             "Magnitude_Acc", "Hor_Acc", "Vert_Acc",
                             "Magnitude_Jerk", "Hor_Jerk", "Vert_Jerk", "NCV"])
        writer.writerow([file_path] + features)

# Extract and save features for each file
for file_path in parkinson_file_list:
    print(f"Processing: {file_path}")
    vel_features = find_velocity(file_path)
    acc_features = find_acceleration(file_path)
    jerk_features = find_jerk(file_path)
    ncv_features = NCV_per_halfcircle(file_path)
    
    # Combine all extracted features into a single list
    combined_features = [
        vel_features[5], vel_features[6], vel_features[7],  # Velocity features
        acc_features[5], acc_features[6], acc_features[7],  # Acceleration features
        jerk_features[4], jerk_features[5], jerk_features[6],  # Jerk features
        ncv_features[1]  # NCV feature
    ]
    
    save_features_to_csv(file_path, combined_features)

print(f"Feature extraction completed! Results saved to {output_csv_path}.")

# Test with Parkinson's data
print('Velocity')
find_velocity(parkinson_file_list[0])
print('Acceleration')
find_acceleration(parkinson_file_list[0])
print('Jerk')
find_jerk(parkinson_file_list[0])
print('NCV')
NCV_per_halfcircle(parkinson_file_list[0])
