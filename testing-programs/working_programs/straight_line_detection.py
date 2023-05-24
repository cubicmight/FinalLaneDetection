import cv2
import numpy as np

# Load the image
img = cv2.imread('michael-images/working-image-1.jfif')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection to detect edges
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Apply HoughLinesP transform to detect lines
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=30, minLineLength=0, maxLineGap=50)

# Maths :(
if lines is not None:
    # Calculate the center of the two longest lines
    longest_lines = sorted(lines, key=lambda x: x[0][2] - x[0][0], reverse=True)[:2]
    x1, y1, x2, y2 = longest_lines[0][0]
    x3, y3, x4, y4 = longest_lines[1][0]
    center_x = int((x1 + x2 + x3 + x4) / 4)
    center_y = int((y1 + y2 + y3 + y4) / 4)
    print("Center line coordinate point 1: ({}, {})".format(center_x, center_y))
    print(x1, x2, y1, y2)
    # Find the slope of one of the lines
    m = (y2 - y1) / (x2 - x1)
    # Find the y-intercept of the center line using the center point as input to determine the y-intercept of the
    # center line since they are parallel. Parallel means same slope different y-intercept.
    c = center_y - (m * center_x)
    # Find the final y values of the two center line points
    final_y1 = int((m * 10000) + c)
    final_y2 = int((m * -10000) + c)
    # Find the final y values of the two arrow line points
    arrow_final_y1 = int((m * 900) + c)
    arrow_final_y2 = int((m * 800) + c)
    # Create the coordinates for the center line points
    p1 = (10000, final_y1)
    p2 = (-10000, final_y2)
    # Create the coordinates for the arrow line points
    pa1 = (900, arrow_final_y1)
    pa2 = (800, arrow_final_y2)

    # Draw a red center point, center line, and direction of travel
    cv2.circle(img, (center_x, center_y), 5, (0, 0, 255), -1)
    cv2.line(img, p1, p2, (255, 0, 255), 2)
    cv2.arrowedLine(img, pa1, pa2, (0, 0, 255), 10)

# Draw the detected lines on the image
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)


# Display the image
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
