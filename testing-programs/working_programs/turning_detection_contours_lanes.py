import os
import cv2
# import matplotlib.pyplot as plt
import numpy as np

###https://flask-bcrypt.readthedocs.io/en/1.0.1/ or https://docs.python.org/3/library/hashlib.html


basedir = os.path.abspath(os.path.dirname(__file__))

image_file_name = "../../videos/solidWhiteRight_test.mp4"
full_image_path = os.path.join(basedir, image_file_name)
if not os.path.exists(full_image_path):
    print("cannot find image")
    exit(-1)

cap = cv2.VideoCapture(full_image_path)

def format_video(cap):
    global output, gray
    output = cap.copy()
    gray = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0);
    gray = cv2.medianBlur(gray, 5)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 3.5)
    kernel = np.ones((3, 3), np.uint8)
    gray = cv2.erode(gray, kernel, iterations=1)
    gray = cv2.dilate(gray, kernel, iterations=1)
    # find the contours from the thresholded image
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    output = cv2.drawContours(output, contours, -1, (0, 255, 0), 2)

    a1 = contours[0][:, 0]
    a2 = contours[1][:, 0]
    # interpolate(a1, a2)


def interpolate(a1, a2, poly_deg=20.5, n_points=100, plot=True):
    min_a1_x, max_a1_x = min(a1[:, 0]), max(a1[:, 0])
    new_a1_x = np.linspace(min_a1_x, max_a1_x, n_points)
    a1_coefs = np.polyfit(a1[:, 0], a1[:, 1], poly_deg)
    new_a1_y = np.polyval(a1_coefs, new_a1_x)
    min_a2_x, max_a2_x = min(a2[:, 0]), max(a2[:, 0])
    new_a2_x = np.linspace(min_a2_x, max_a2_x, n_points)
    a2_coefs = np.polyfit(a2[:, 0], a2[:, 1], poly_deg)
    new_a2_y = np.polyval(a2_coefs, new_a2_x)
    midx = [np.mean([new_a1_x[i], new_a2_x[i]]) for i in range(n_points)]
    midy = [np.mean([new_a1_y[i], new_a2_y[i]]) for i in range(n_points)]
    # if plot:
        # plt.plot(midx, midy, '--', c='black')

    return np.array([[x, y] for x, y in zip(midx, midy)])



while True:
    ret, frame = cap.read()
    if not ret:
        exit(-1)
    format_video(frame)
    cv2.imshow('gray', gray)
    cv2.imshow('frame', output)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break







