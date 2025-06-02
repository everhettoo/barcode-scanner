"""
This module provides the shape related functionality.
"""
import cv2
import numpy as np

from ipcv import imutil
from ipcv.cvlib import resize_box


def find_rectangle(processed_img, min_area_factor, cnt, box=False, draw=False, verbose=False):
    """
    Finds a rectangle in the image and returns it's coordinates.
    :param processed_img: The processed image.
    :param min_area_factor: Minimum area to consider a rectangle.
    :param cnt: The current parent iteration count.
    :param box: Is the rectangle a box (aspect ratio 1:2).
    :param draw: For debugging purposes - draws rectangles.
    :param verbose: For debugging purposes - prints debugging information.
    :return: The coordinates of the rectangle.
    """
    # TODO: RETR_EXTERNAL (as disabled below) - does not return inner rect that meets the requirement.
    # contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(processed_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    image_area = processed_img.shape[0] * processed_img.shape[1]
    min_required_area = image_area * min_area_factor
    max_required_area = image_area - (image_area * 0.3)
    print(
        f'[{cnt}] --Contour       : found={len(contours)}; min-req-area={min_required_area:,.2f}, max-req-area={max_required_area:,.2f} '
        f'({min_area_factor} at rate); area should be > {(min_required_area / image_area) * 100:.2f}%')

    selected_max_contour = None
    selected_max_area = 0

    # Epsilon: is maximum distance from contour to approximated contour.
    # eps = 0.04
    eps = 0.0391
    # eps = 0.05

    curr_cnt = 1
    # Contour has coordinates for drawing the detected shape (polygons - can be more than 3 or 4)
    for contour in contours:
        # Contour-perimeter: calculates the perimeter for the given points when it's a closed lines.
        perimeter = cv2.arcLength(contour, True)

        # Ramer–Douglas–Peucker algorithm: It approximates a contour shape to another shape with reduced vertices
        # depending upon the precision of epsilon. So, coordinates for polygons are returned.
        approx = cv2.approxPolyDP(contour, eps * perimeter, True)
        if draw: cv2.drawContours(processed_img, [approx], -1, imutil.BLUE_COLOR, 3)

        # Only polygons with 4 angles are considered since a box or rectangle is needed.
        if len(approx) == 4:
            calculated_area = cv2.contourArea(contour)

            # Retrieve the four coordinates (of a most-likely rectangle) to verify if it is a rectangle.
            [a, b, c, d] = imutil.rectangle_coordinates(approx)
            if verbose:
                print(f'[c:{curr_cnt}] --Contour     : calculated area={calculated_area:,.2f}, '
                      f'coordinates (X,Y)=[A=({a}), B=({b}), C=({c}), D=({d})]')

            if (imutil.is_perpendicular_angle(imutil.calculate_angle(a, b, c))
                    and imutil.is_perpendicular_angle(imutil.calculate_angle(b, c, d))
                    and imutil.is_perpendicular_angle(imutil.calculate_angle(c, d, a))
                    and imutil.is_perpendicular_angle(imutil.calculate_angle(d, a, b))):

                (x, y, w, h) = cv2.boundingRect(approx)
                calculated_aspect_ratio = w / float(h)
                if verbose:
                    print(f'[c:{curr_cnt}] --Contour     : calculated ratio={calculated_aspect_ratio:,.2f}')

                # Selecting the coordinates that fulfills min required area for box and rectangle.
                if box:
                    if calculated_area > min_required_area and 0.8 <= calculated_aspect_ratio <= 1.2:
                        if calculated_area > selected_max_area:
                            selected_max_area = calculated_area
                            print(
                                f'[c:{curr_cnt}] --Contour(box) : selected max-area={selected_max_area:,.2f} '
                                f'at rate={(selected_max_area / image_area) * 100:.2f}%')
                            selected_max_contour = contour
                else:
                    # Define the rectangle's criteria.
                    if min_required_area < calculated_area < max_required_area:
                        if calculated_area > selected_max_area:
                            # if 1.2 < calculated_aspect_ratio < 3:
                            if calculated_aspect_ratio > 1.2 and calculated_aspect_ratio < 6:
                                selected_max_area = calculated_area
                                print(
                                    f'[c:{curr_cnt}] --Contour     : selected max-area={selected_max_area:,.2f} '
                                    f'at rate={(selected_max_area / image_area) * 100:.2f}, aspect-ratio={calculated_aspect_ratio:.6f}')
                                selected_max_contour = contour
            curr_cnt += 1
            if verbose:
                print('\r')
    return selected_max_contour


def get_prominent_contour(processed_image, offset=0):
    """
    Gets the prominent contour from the given processed image and draws a box on the source image.
    :param processed_image: The processed image to trace the biggest contour on.
    :param offset: The offset of the box to resize to.
    :return: Returns the prominent contour.
    """
    contours, hierarchy = cv2.findContours(processed_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)
    box = np.intp(cv2.boxPoints(rect))

    # Increase the box offset for better detection.
    box = resize_box(box, offset)

    return box
