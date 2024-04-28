'''
STEP 3 part1
Use Kalman filter to estimate positions for the lost targets set as -1.
It works well when only a few points are lost.
'''

import numpy as np
import pandas as pd

csv_path = 'csv/video1_car.csv'
csv_kalman_path = 'csv/video1_car_kalman.csv'

# The matrix tracks x center and y center, so we only need 4x4 matrix.
# The box size for lost detections are set as the w and h of the last detected box.
A = np.array([[1,0,1,0],
              [0,1,0,1],
              [0,0,1,0],
              [0,0,0,1]])

# If target is lost
A_ = np.array([[1,0,1,0],
               [0,1,0,1],
               [0,0,1,0],
               [0,0,0,1]])

H = np.eye(4) # State observation matrix
Q = np.eye(4) * 0.1 # Process noise covariance matrix Q, p(w)~N(0,Q), noise from uncertainties in the real world,
R = np.eye(4) * 10 # Observation noise covariance matrix R, p(v)~N(0,R)
B = None # Control input matrix B
P = np.eye(4) # Initialization of state estimate covariance matrix P

def get_xyxy(X, w, h):
    xcen, ycen, dx, dy = X[0], X[1], X[2], X[3]
    xmin = xcen - w/2
    ymin = ycen - h/2
    xmax = xcen + w/2
    ymax = ycen + h/2
    return np.array([xmin, ymin, xmax, ymax])

def get_wh(X):
    xmin, ymin, xmax, ymax = X
    w = xmax - xmin
    h = ymax - ymin
    return w, h

def get_box_center(X):
    xmin, ymin, xmax, ymax = X
    xcen = (xmin + xmax) / 2
    ycen = (ymin + ymax) / 2
    return np.array([xcen, ycen])

# ---------------- START -----------------

df = pd.read_csv(csv_path)

initial_target_box = []
initial_state = []

# Set the first detected position of the target as the initial position of tracking box
for index, row in df.iterrows():
    if row.xmin != -1 :
        initial_target_box = [row.xmin, row.ymin, row.xmax, row.ymax]
        center = get_box_center(initial_target_box)
        initial_state = [center[0], center[1], 0, 0]  # x_cen, y_cen, dx, dy
        print(center)
        break

X_posterior = np.array(initial_state)
P_posterior = np.array(P)
Z = np.array(initial_state)
w, h = 2, 2

trace_list = [] # tracking dots positions
truth_list = [] # real target positions
xmin_list = []
ymin_list = []
xmax_list = []
ymax_list = []

for index, row in df.iterrows():
    print(f'----------------------- LINE {index} START -------------------------')
    targetFound = False
    box = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
    box_center = get_box_center(box)
    truth_list.append(box_center)

    if row['xmin'] != -1:
        targetFound = True

    # Get observation Z
    if targetFound:
        w, h = get_wh(box)
        last_box_posterior = get_xyxy(X_posterior, w, h)
        box_center = get_box_center(box)
        Z[0], Z[1] = box_center[0], box_center[1]

        dx = box_center[0] - X_posterior[0]
        dy = box_center[1] - X_posterior[1]
        Z[2], Z[3] = dx, dy
        print('Z:', Z)

    # KALMAN TRACKER
    if targetFound:
        # 1. Get prior estimation
        X_prior = np.dot(A, X_posterior)
        print("X_prior: ", X_prior)
        box_prior = get_xyxy(X_prior, w, h)

        # 2. Get covariance matrix of the state estimate P
        P_prior_1 = np.dot(A, P_posterior)
        P_prior = np.dot(P_prior_1, A.T) + Q
        # print("P_prior: \n", P_prior)

        # 3. Get Kalman gain
        k1 = np.dot(P_prior, H.T)
        k2 = np.dot(np.dot(H, P_prior), H.T) + R
        K = np.dot(k1, np.linalg.inv(k2))

        # 4. Get posterior estimation
        X_posterior_1 = Z - np.dot(H, X_prior)
        X_posterior = X_prior + np.dot(K, X_posterior_1)
        print("X_posterior: ", X_posterior)
        box_posterior = get_xyxy(X_posterior, w, h)

        # 5. Update covariance matrix of the state estimate P
        P_posterior_1 = np.eye(4) - np.dot(K, H)
        P_posterior = np.dot(P_posterior_1, P_prior)

    # If target is lost, the previous estimate is used as the prior estimate.
    else:
        X_posterior = np.dot(A_, X_posterior)
        print("X_posterior: ", X_posterior)
        box_posterior = get_xyxy(X_posterior, w, h)

    # Update dataframe
    box_center = get_box_center(box_posterior)
    trace_list.append(box_center)

    xmin_list.append(box_posterior[0])
    ymin_list.append(box_posterior[1])
    xmax_list.append(box_posterior[2])
    ymax_list.append(box_posterior[3])

df['xmin'] = xmin_list
df['ymin'] = ymin_list
df['xmax'] = xmax_list
df['ymax'] = ymax_list
x_cen = [point[0] for point in trace_list]
y_cen = [point[1] for point in trace_list]
df['x_cen'] = x_cen
df['y_cen'] = y_cen

df.to_csv(csv_kalman_path , index=False)
