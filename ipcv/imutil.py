"""
This module provides common required functions related to image processing.
"""
import math

import cv2
import numpy as np

BLUE_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 255, 0)
RED_COLOR = (255, 0, 0)
BLACK_COLOR = 0
WHITE_COLOR = 255


def calculate_angle(point_a, point_b, point_c):
    """
    Calculates the angle formed by three points.
    :param point_a: A tuple or list representing the coordinates of point A (x1, y1).
    :param point_b: A tuple or list representing the coordinates of point B (x2, y2).
    :param point_c: A tuple or list representing the coordinates of point C (x3, y3).
    :return: The angle in degrees.
    """
    x1, y1 = point_a
    x2, y2 = point_b
    x3, y3 = point_c

    # Create vectors
    vector_ba = (x1 - x2, y1 - y2)
    vector_bc = (x3 - x2, y3 - y2)

    # Calculate the dot product
    dot_product = vector_ba[0] * vector_bc[0] + vector_ba[1] * vector_bc[1]

    # Calculate the magnitudes
    magnitude_ba = math.sqrt(vector_ba[0] ** 2 + vector_ba[1] ** 2)
    magnitude_bc = math.sqrt(vector_bc[0] ** 2 + vector_bc[1] ** 2)

    # Calculate the cosine of the angle
    cos_theta = dot_product / (magnitude_ba * magnitude_bc)

    # Calculate the angle in radians
    angle_rad = math.acos(cos_theta)

    # Convert the angle to degrees
    angle_deg = math.degrees(angle_rad)

    return angle_deg


def crop_roi(image, max_contour, color):
    perimeter = cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, 0.04 * perimeter, True)
    cv2.drawContours(image, [approx], -1, color, 3)
    (x, y, w, h) = cv2.boundingRect(approx)
    return image[y:y + h, x:x + w]


def draw_bounding_box(image, contour, color):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    cv2.drawContours(image, [approx], -1, color, 3)


def draw_bounding_box2(image, box, color):
    cv2.drawContours(image, [box], -1, color, 3)


def crop_roi2(image, box):
    (x, y, w, h) = cv2.boundingRect(box)
    return image[y:y + h, x:x + w]


def is_perpendicular_angle(angle):
    if abs(90 - angle) < 10:
        return True
    else:
        return False


def rectangle_coordinates(approx):
    """
    Reshapes the approx coordinates of a rectangle inside a list-of-list to flat-list and returns the four coordinates.
    :param approx: Coordinates of a most-likely rectangle.
    :return: Four coordinates.
    """
    coordinates = approx.reshape(4, 2)
    return coordinates[0], coordinates[1], coordinates[2], coordinates[3]


def pixel_percentage(image, color):
    return np.sum(image == color) / (image.shape[0] * image.shape[1]) * 100
