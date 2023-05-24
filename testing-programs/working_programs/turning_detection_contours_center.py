import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("../turning_pictures/testing_images/01. solidWhiteCurve.jpeg")
# convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# create a binary thresholded image
_, binary = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
# find the contours from the thresholded image
contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

a1 = contours[0][:, 0]
a2 = contours[1][:, 0]


def interpolate(a1, a2, poly_deg=20.5, n_points=100, plot=True):

    min_a1_x, max_a1_x = min(a1[:,0]), max(a1[:,0])
    new_a1_x = np.linspace(min_a1_x, max_a1_x, n_points)
    a1_coefs = np.polyfit(a1[:,0],a1[:,1], poly_deg)
    new_a1_y = np.polyval(a1_coefs, new_a1_x)

    min_a2_x, max_a2_x = min(a2[:,0]), max(a2[:,0])
    new_a2_x = np.linspace(min_a2_x, max_a2_x, n_points)
    a2_coefs = np.polyfit(a2[:,0],a2[:,1], poly_deg)
    new_a2_y = np.polyval(a2_coefs, new_a2_x)

    midx = [np.mean([new_a1_x[i], new_a2_x[i]]) for i in range(n_points)]
    midy = [np.mean([new_a1_y[i], new_a2_y[i]]) for i in range(n_points)]

    if plot:
        plt.plot(midx, midy, '--', c='black')

    return np.array([[x, y] for x, y in zip(midx, midy)])

interpolate(a1, a2)
plt.imshow(image)
plt.show()
