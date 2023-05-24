def grey(image):
    # convert to grayscale
    image = np.asarray(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


# Apply Gaussian Blur --> Reduce noise and smoothen image
def gauss(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


# outline the strongest gradients in the image --> this is where lines in the image are
def canny(image):
    edges = cv2.Canny(image, 50, 150)
    return edges


def region(image):
    height, width = image.shape
    # isolate the gradients that correspond to the lane lines
    triangle = np.array([
        [(100, height), (475, 325), (width, height)]
    ])
    # create a black image with the same dimensions as original image
    mask = np.zeros_like(image)
    # create a mask (triangle that isolates the region of interest in our image)
    mask = cv2.fillPoly(mask, triangle, 255)
    mask = cv2.bitwise_and(image, mask)
    return mask


def display_lines(image, lines):
    lines_image = np.zeros_like(image)
    # make sure array isn't empty
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            # draw lines on a black image
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return lines_image


def average(image, lines):
    left = []
    right = []

    if lines is not None:
        for line in lines:
            print(line)
            x1, y1, x2, y2 = line.reshape(4)
            # fit line to points, return slope and y-int
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            print(parameters)
            slope = parameters[0]
            y_int = parameters[1]
            # lines on the right have positive slope, and lines on the left have neg slope
            if slope < 0:
                left.append((slope, y_int))
            else:
                right.append((slope, y_int))

    # takes average among all the columns (column0: slope, column1: y_int)
    right_avg = np.average(right, axis=0)
    left_avg = np.average(left, axis=0)
    # create lines based on averages calculates
    left_line = make_points(image, left_avg)
    right_line = make_points(image, right_avg)
    return np.array([left_line, right_line])


def make_points(image, average):
    print(average)
    try:
        slope, y_int = average
    except TypeError:
        slope, y_int = 0.001, 0
    y1 = image.shape[0]
    # how long we want our lines to be --> 3/5 the size of the image
    y2 = int(y1 * (3 / 5))
    # determine algebraically
    # if math.isinf(int(slope)) == False:
    #     x1 = int((y1 - y_int) // slope)
    #     x2 = int((y2 - y_int) // slope)
    #     return np.array([x1, y1, x2, y2])
    # else:
    #     pass

    x1 = int((y1 - y_int) // slope)
    x2 = int((y2 - y_int) // slope)
    return np.array([x1, y1, x2, y2])




'''##### DETECTING lane lines in image ######'''


def detect_lanes(image1):
    ##filter for longest line
    #create horizontal line between lines at the bottom and top
    #find center of the horizontal line
    #draw line through there
    #do for each frame
    copy = np.copy(image1)
    edges = cv2.Canny(copy, 50, 150)
    isolated = region(edges)
    # cv2.imshow("edges", edges)
    # cv2.imshow("isolated", isolated)
    # cv2.waitKey(0)
    # DRAWING LINES: (order of params) --> region of interest, bin size (P, theta), min intersections needed, placeholder array,
    lines = cv2.HoughLinesP(isolated, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average(copy, lines)
    black_lines = display_lines(copy, averaged_lines)
    # taking wighted sum of original image and lane lines image
    lanes = cv2.addWeighted(copy, 0.8, black_lines, 1, 1)
    print('frame')
    return lanes
    # cv2.imshow("lanes", lanes)
    # cv2.waitKey(0)


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
    return output