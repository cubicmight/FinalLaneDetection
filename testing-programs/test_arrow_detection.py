import cv2
import numpy as np

import cv2
import numpy as np

# Load the arrow template image
template = cv2.imread('test_images/img_3.png', 0)  # Replace 'arrow_template.png' with the actual template image file

# Load the road image
road_image = cv2.imread('test_images/img_3.png')  # Replace 'road_image.png' with the actual road image file

# Resize the template image to match the dimensions of the road image
template = cv2.resize(template, (road_image.shape[1], road_image.shape[0]))

# Convert the road image to grayscale
gray_image = cv2.cvtColor(road_image, cv2.COLOR_BGR2GRAY)

# Perform template matching
result = cv2.matchTemplate(gray_image, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.8  # Adjust the threshold as needed
locations = np.where(result >= threshold)

# Iterate over the matched locations
for loc in zip(*locations[::-1]):
    # Extract the arrow bounding box
    x, y = loc
    w, h = template.shape[::-1]
    arrow_roi = road_image[y:y+h, x:x+w]

    # Perform arrow direction detection
    arrow_gray = cv2.cvtColor(arrow_roi, cv2.COLOR_BGR2GRAY)
    _, arrow_thresh = cv2.threshold(arrow_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(arrow_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate over the contours and find the arrow's direction
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Adjust the minimum contour area as needed
            # Calculate the centroid of the contour
            M = cv2.moments(contour)
            centroid_x = int(M['m10'] / M['m00'])
            centroid_y = int(M['m01'] / M['m00'])

            # Determine the arrow's direction based on the centroid position
            arrow_direction = 'up'  # Assume the arrow is pointing up
            if centroid_x < w / 3:
                arrow_direction = 'left'
            elif centroid_x > 2 * w / 3:
                arrow_direction = 'right'
            elif centroid_y > 2 * h / 3:
                arrow_direction = 'down'

            # Draw the arrow and its direction on the road image
            cv2.drawContours(road_image, [contour], 0, (0, 255, 0), 2)
            cv2.putText(road_image, arrow_direction, (centroid_x, centroid_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# Display the road image with detected arrows
cv2.imshow('Detected Arrows', road_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
