import cv2
import numpy as np

# Read image
image = cv2.imread('turning_pictures/draw.io_curved_image_1.png')

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use canny edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Apply HoughLinesP method to
# to directly obtain line end points
lines_list = []
lines = cv2.HoughLinesP(
    edges,  # Input edge image
    1,  # Distance resolution in pixels
    np.pi / 180,  # Angle resolution in radians
    threshold=5,  # Min number of votes for valid line
    minLineLength=0,  # Min allowed length of line
    maxLineGap=0  # Max allowed gap between line for joining them
)

# Iterate over points
for points in lines:
    # Extracted points nested in the list
    x1, y1, x2, y2 = points[0]
    # Draw the lines join the points on the original image
    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
    # cv2.putText(image, "(%d, %d)" % (x1, y1), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    # cv2.putText(image, "(%d, %d)" % (x2, y2), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    # Maintain a simples lookup list for points
    lines_list.append([(x1, y1), (x2, y2)])

print(lines_list)
# Calculate the total sum of x-values and y-values
# sum_x = 0
# sum_y = 0
# count = 0
#
# for pair in lines_list:
#     for point in pair:
#         sum_x += point[0]
#         sum_y += point[1]
#         count += 1
#
# # Calculate the average of x-values and y-values
# avg_x = sum_x / count
# avg_y = sum_y / count
# center_coordinates = (avg_x, avg_y)
# # Print the average coordinates
# print(f"The average x-coordinate is {avg_x:.2f}")
# print(f"The average y-coordinate is {avg_y:.2f}")
#
# cv2.circle(image, (891,281), 5, (0, 0, 255), 5)

img_resized = cv2.resize(image, (1000, 1000))
cv2.imshow("Curves Detected With Midline", img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
