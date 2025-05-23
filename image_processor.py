import math

import cv2
import numpy as np


def detect_barcode(image):
    # Convert to grayscale for processing.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find vertical lines (x-axis intensity change when y = 0).
    gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)

    # Find horizontal lines (y-axis intensity change when x = 0).
    gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)

    # Converts negative values to absolute values |x|.
    gradient = cv2.convertScaleAbs(gradient)

    # blur and threshold the image
    blurred = cv2.blur(gradient, (3, 3))
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

    # construct a closing kernel and apply it to the thresholded image
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, se)

    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    # find the contours in the thresholded image, then sort the contours by their area, keeping only the largest one
    (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = np.intp(cv2.boxPoints(rect))
    # draw a bounding box arounded the detected barcode and display the
    # image
    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

    [X, Y, W, H] = cv2.boundingRect(box)

    cropped = image[Y:Y + H, X:X + W]

    return image, cropped


def detect_barcode2(image):
    # Convert to grayscale for processing.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gamma_corrected = adjust_gamma(gray, 0.8)

    blurred = gaussian_blur(gamma_corrected, (15, 15))

    boosted = high_boost_filter(blurred, (9, 9))

    gradient = detect_gradient(boosted)

    thresh = blur_threshold(gradient, [3, 3])

    morphed = dilate(thresh, [21, 7], 4)

    # Shrink
    new_width = int(morphed.shape[1] / 6)
    new_height = int(morphed.shape[0] / 6)

    shrunk = resize_image(morphed, new_width, new_height)

    enlarged = resize_image(shrunk, morphed.shape[1], morphed.shape[0])

    # find the contours in the thresholded image, then sort the contours by their area, keeping only the largest one
    (cnts, _) = cv2.findContours(enlarged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = np.intp(cv2.boxPoints(rect))
    # draw a bounding box arounded the detected barcode and display the image
    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)

    [X, Y, W, H] = cv2.boundingRect(box)
    cropped = image[Y:Y + H, X:X + W]

    return image, cropped


# Increase contrast
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def gaussian_blur(image, ksize):
    return cv2.GaussianBlur(image, ksize, 0)


# Detect edges (canny no good here!)
def high_boost_filter(image, ksize):
    mean = np.ones(ksize, np.float32) / math.prod(ksize)
    blur = cv2.filter2D(image, -1, mean)
    # edges = image - blur
    # Subtract the y-gradient from the x-gradient
    edges = cv2.subtract(image, blur)
    edges = cv2.convertScaleAbs(edges)

    return image + edges


def detect_gradient(image):
    # Find vertical line intensity change (using x-axis, when y = 0).
    grad_x = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)

    # Find horizontal line intensity change(using y-axis, when x = 0).
    grad_y = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

    # Subtract the y-gradient from the x-gradient
    grad = cv2.subtract(grad_x, grad_y)

    # Converts negative values to absolute values |x|.
    return cv2.convertScaleAbs(grad)


def blur_threshold(image, ksize):
    blurred = cv2.blur(image, ksize=ksize)
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
    return thresh


def dilate(image, ksize, iteration):
    se = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, se)

    # perform a series of erosions and dilations
    # closed = cv2.erode(closed, None, iterations=iteration)
    closed = cv2.dilate(closed, None, iterations=iteration)
    return closed


def load_image(file_path):
    # IMREAD_UNCHANGED - loads alpha channel.
    # img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    # Convert the image from BGR to RGB
    return cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)


def resize_image(image, width, height):
    img = cv2.resize(image, (width, height))
    return img


class ImageProcessor:
    def __init__(self, image):
        self.image = image
