import cv2
import numpy as np


def region_of_interest(img, vertices):
    """
    Applies a region of interest mask to the image.
    """
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def draw_lines(img, lines, color=(0, 255, 0), thickness=3):
    """
    Draws lines on the image.
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def detect_lanes(img):
    """
    Detects lanes in the image.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using the Canny algorithm
    edges = cv2.Canny(blur, 50, 150)

    # Define a region of interest (ROI) for lane detection
    height, width = edges.shape[:2]
    roi_vertices = np.array([[(0, height), (width / 2, height / 2), (width, height)]], dtype=np.int32)
    roi = region_of_interest(edges, roi_vertices)

    # Perform Hough line detection to find lane lines
    lines = cv2.HoughLinesP(roi, rho=1, theta=np.pi / 180, threshold=30, minLineLength=20, maxLineGap=100)

    # Create an empty image to draw the lane lines
    lane_lines_img = np.zeros_like(img)

    # Calculate the center line position
    center_x = width / 2

    # Define the maximum y-coordinate for the lane lines to be drawn
    max_y = int(height * 0.6)

    # Draw the lane lines
    left_lane_lines = []
    right_lane_lines = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            # Calculate the slope of the line
            slope = (y2 - y1) / (x2 - x1)

            # Classify the line as left or right based on the slope and its position relative to the center
            if slope < -0.2 and x1 < center_x and x2 < center_x:
                left_lane_lines.append(line)
            elif slope > 0.2 and x1 > center_x and x2 > center_x:
                right_lane_lines.append(line)

    # Draw the left lane lines
    draw_lines(lane_lines_img, left_lane_lines, color=(0, 0, 255))

    # Draw the right lane lines
    draw_lines(lane_lines_img, right_lane_lines, color=(0, 0, 255))

    # Draw the center line up to the maximum y-coordinate
    cv2.line(lane_lines_img, (int(center_x), height), (int(center_x), max_y), color=(0, 255, 0), thickness=3)

    # Combine the lane lines image with the original image
    output = cv2.addWeighted(img, 0.8, lane_lines_img, 1, 0)

    return output


# Load video file
video_path = '../videos/5-21-23-8385-test1.mp4'
cap = cv2.VideoCapture(video_path)

# Loop through each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect lanes in the frame
    lanes_frame = detect_lanes(frame)

    # Display the resulting frame
    cv2.imshow('Lane Detection', lanes_frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
