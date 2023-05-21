import cv2
from canny import *

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    return image & mask

def get_slope_intercepts(lines):
    slopes = (lines[:, 0, 3] - lines[:, 0, 1]) / (lines[:, 0, 2] - lines[:, 0, 0])  # (y2-y1)/(x2-1)
    intercepts = lines[:, 0, 1] - (slopes * lines[:, 0, 0])                         # (b = y - mx)
    avg_positive_intercepts, avg_negative_intercepts = np.average(intercepts[slopes >= 0]), np.average(intercepts[slopes < 0])
    avg_positive_slope, avg_negative_slope = np.average(slopes[slopes >= 0]), np.average(slopes[slopes < 0])
    return (avg_negative_slope, avg_negative_intercepts, avg_positive_slope, avg_positive_intercepts)

def draw_line(image, slope, intercept, length):
    d = length
    m = slope
    b = intercept
    y1 = image.shape[0]
    y2 = y1 - (np.abs(m) * d) / np.sqrt(np.square(m) + 1)
    x1 = (y1 - b) / m
    x2 = (y2 - b) / m
    if x1 > 0 and y1 > 0 and x2 < image.shape[1] and y2 < image.shape[0]:
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 10)

def get_lines(image, lines):
    length = 500
    line_image = np.zeros_like(image)
    m_left, b_left, m_right, b_right = get_slope_intercepts(lines)
    draw_line(line_image, m_left, b_left, length)
    draw_line(line_image, m_right, b_right, length)
    return line_image

def process_frame(image):
    canny = get_canny(image)
    cropped = region_of_interest(canny)
    lines = cv2.HoughLinesP(cropped, 2, np.deg2rad(1), 100, np.array([]), minLineLength=40, maxLineGap=5)
    line_image = np.repeat(get_lines(canny, lines)[:,:,np.newaxis], 3, axis=2)
    overlayed_image = image | line_image
    composite_image_top = np.append(canny, cropped, axis=1)
    composite_image_top = np.repeat(composite_image_top[:,:,np.newaxis], 3, axis=2)
    composite_image_bottom = np.append(line_image, overlayed_image, axis=1)
    composite_image = np.append(composite_image_top, composite_image_bottom, axis=0)
    return composite_image

video = cv2.VideoCapture("test2.mp4")
cv2.namedWindow("lanes", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("lanes", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
while (video.isOpened()):
    _, frame = video.read()
    if frame is not None:
        processed_frame = process_frame(frame)
        cv2.imshow("lanes", processed_frame)
        if cv2.waitKey(1) == ord(' '):
            break
video.release()
cv2.destroyAllWindows()