"""
This module provides the barcode and qrcode scan functionality.
:param kwargs (used in this module):
- image (matrix): Image in which the barcode needs to be detected.
- gamma (float): A number to set the gamma value.
- gaussian_ksize (tuple): The gaussian kernel size used for smoothing an image. E.g. (3,3) or (9,9).
- gaussian_sigma (float): The gaussian_sigma used for smoothing an image.
- avg_ksize1 (tuple): The kernel size used for the first average smoothing. E.g. (3,3) or (9,9).
- avg_ksize2 (tuple): The kernel size used for the second average smoothing. E.g. (3,3) or (9,9).
- thresh_min (uint): The threshold value for binarizing an image.
- dilate_kernel (tuple): The kernel size of structuring element used for dilation. E.g. (21,7) or (51,9).
- dilate_iteration (uint): The number of interation the dilation is performed.
- shrink_factor (uint):  The original image size is divided with shrink_factor to resize an image (shrinking).
- offset (uint):  The offest of the box to resize to.
"""
import threading
from math import ceil

import cv2
import numpy as np

from ipcv import cvlib, imutil, shape

MIN_THRESHOLD_LIMIT = 220

calculate_threshold = lambda t, i, d: ceil(t + (t * i * d))


def preprocess_image(image, gamma, gaussian_ksize, gaussian_sigma):
    """
    Preprocesses the given image for code detection.
    :param image: The image to preprocess.
    :param gamma: The gamma value.
    :param gaussian_ksize: The kernel size used for gaussian smoothing.
    :param gaussian_sigma: The gaussian_sigma used for smoothing an image.
    :return: The processed image.
    """
    # Adjust the contrast
    p = cvlib.adjust_gamma(image, gamma)

    # Convert image to gray scale for processing.
    p = cvlib.convert_rgb2gray(p)

    # Remove noise.
    p = cvlib.gaussian_blur(p, gaussian_ksize, gaussian_sigma)

    return p


def preprocess_image2(image, gamma, gaussian_ksize, gaussian_sigma):
    """
    Preprocesses the given image for code detection.
    :param image: The image to preprocess.
    :param gamma: The gamma value.
    :param gaussian_ksize: The kernel size used for gaussian smoothing.
    :param gaussian_sigma: The gaussian_sigma used for smoothing an image.
    :return: The processed image.
    """
    # Convert image to gray scale for processing.
    p = cvlib.convert_rgb2gray(image)

    # Adjust the contrast
    p = cvlib.adjust_gamma(p, gamma)

    # Remove noise.
    p = cvlib.gaussian_blur(p, gaussian_ksize, gaussian_sigma)

    return p


def adjust_threshold(i, threshold_min, rate, black_pixels, white_pixels, max_pixel_limit):
    """
    Control logic used by detect_barcode v2.
    :param i: The iteration count.
    :param threshold_min: As documented in the module section.
    :param rate: As documented in the module section.
    :param black_pixels: The black pixels count.
    :param white_pixels: The white pixels count.
    :param max_pixel_limit: As documented in the module section.
    :return:
    """
    if threshold_min > MIN_THRESHOLD_LIMIT:
        return MIN_THRESHOLD_LIMIT, True
    elif black_pixels > max_pixel_limit or white_pixels > max_pixel_limit:
        if white_pixels > max_pixel_limit:
            # Reduce 20%
            thresh = calculate_threshold(threshold_min, i, rate)
            thresh = thresh - (thresh * 0.2)
            return thresh, True
        else:
            # Increase 20%
            thresh = calculate_threshold(threshold_min, i, rate)
            thresh = thresh + (thresh * 0.2)
            return thresh, True
    else:
        # thresh = ceil(threshold_min + (threshold_min * i * rate))
        thresh = calculate_threshold(threshold_min, i, rate)
        if thresh > MIN_THRESHOLD_LIMIT:
            return MIN_THRESHOLD_LIMIT, True
        else:
            return thresh, False


def detect_barcode(**kwargs):
    """
    Processes the given image to detect barcode using the parameters documented in the module section.
    :return: The detected rectangle's coordinates.
    """
    p = cvlib.convert_rgb2gray(kwargs['image'])

    p = cvlib.adjust_gamma(p, kwargs['gamma'])

    p = cvlib.gaussian_blur(p, kwargs['gaussian_ksize'], kwargs['gaussian_sigma'])

    p = cvlib.average_blur(p, kwargs['avg_ksize1'])

    p = cvlib.detect_gradient(p)

    # Only [3,3] works
    p = cvlib.average_blur(p, kwargs['avg_ksize2'])

    p = cvlib.binarize(p, kwargs['thresh_min'])

    # p = cvlib.dilate(p, kwargs['dilate_kernel'], kwargs['dilate_iteration'])
    p = cvlib.morph_close(p, kwargs['dilate_kernel'])
    p = cvlib.morph_dilate(p, kwargs['dilate_iteration'])

    # Record the image's original size for enlarging.
    x = p
    new_width = int(p.shape[1] / kwargs['shrink_factor'])
    new_height = int(p.shape[0] / kwargs['shrink_factor'])

    p = cvlib.resize_image(p, new_width, new_height)

    p = cvlib.resize_image(p, x.shape[1], x.shape[0])

    contour = shape.get_prominent_contour(p, kwargs['offset'])

    return contour


def detect_barcode_v2(image, **kwargs):
    """
    Processes the given image to detect barcode using the parameters documented in the module section.
    :return: The detected rectangle's coordinates.
    """
    max_pixel_limit = int(kwargs['max_pixel_limit'])
    min_threshold = int(kwargs['min_threshold'])
    cropped = None
    thresh_exceeded = False
    cnt = 0
    # To prevent repetition of warning.
    displayed_warning = False
    # Assign current-threshold for processing with config-threshold.
    curr_threshold = int(kwargs['min_threshold'])
    # Current processed image.
    p = None

    pre = preprocess_image(image, kwargs['gamma'], kwargs['gaussian_ksize'], kwargs['gaussian_sigma'])
    image_size = image.shape[0] * image.shape[1]
    print(f'Info                : size={image_size:,}, RxC=[{pre.shape[0]:,}x{pre.shape[1]:,}], '
          f'box-ratio-on: {kwargs["box"]}, attempt-limit: {kwargs["attempt_limit"]}')

    for i in range(1, int(kwargs['attempt_limit']) + 1):
        cnt = i
        print(f'----------> [BEGIN, attempt={cnt}]')
        if not thresh_exceeded:
            # Binarize image using the threshold (initially from config, subsequently using calculated).
            p = cvlib.binarize_inv(pre, curr_threshold)

            # Get pixel values in % after binarization.
            black_pixels = imutil.pixel_percentage(p, imutil.BLACK_COLOR)
            white_pixels = imutil.pixel_percentage(p, imutil.WHITE_COLOR)

            # No logic processing, only for displaying.
            if black_pixels > max_pixel_limit or white_pixels > max_pixel_limit:
                print(
                    f'{[i]} --Binarize      : pixel ratio exceeded limit {max_pixel_limit:}% with min-threshold={curr_threshold}!'
                    f' [black={black_pixels:,.2f}%, white={white_pixels:,.2f}%]')
            elif curr_threshold > MIN_THRESHOLD_LIMIT:
                print(
                    f'{[i]} --Binarize      : min-threshold={curr_threshold} exceeded limit ({MIN_THRESHOLD_LIMIT})!\n'
                    f'[black={black_pixels:,.2f}%, white={white_pixels:,.2f}%].')
            else:
                print(f'{[i]} --Binarize      : at min-thresh={curr_threshold:,}, '
                      f'[black={black_pixels:,.2f}%, white={white_pixels:,.2f}%]')

            # Pass the config-threshold to obtain the calculated current-threshold for processing.
            # On breaching threshold limit, revised threshold is made for correction before flagging threshold off.
            curr_threshold, thresh_exceeded = adjust_threshold(i,
                                                               int(kwargs['min_threshold']),
                                                               kwargs["threshold_rate"],
                                                               black_pixels,
                                                               white_pixels,
                                                               max_pixel_limit)

            if thresh_exceeded:
                calculated_limit = calculate_threshold(min_threshold, i - 1, kwargs["threshold_rate"])
                # Display the descriptive warning message once.
                if not displayed_warning:
                    if curr_threshold == MIN_THRESHOLD_LIMIT:
                        print(
                            f'{[i]} --Binarize      : min-threshold={calculated_limit} exceeded limit ({MIN_THRESHOLD_LIMIT})!\n'
                            f'                      indefinite-adjustment made to min-threshold={curr_threshold:} ...')
                    else:
                        print(
                            f'{[i]} --Binarize      : pixel ratio exceeded limit {max_pixel_limit:}% with min-threshold={calculated_limit}!\n'
                            f'                      indefinite-adjustment made to min-threshold={curr_threshold:} ...')

                    # Turn off to avoid repeating warning.
                    displayed_warning = True

                # Binarize image using corrected threshold before turning off binarization.
                print(f'{[i]} --Binarize      : reverting image with corrected min-thresh={curr_threshold:,}')
                p = cvlib.binarize_inv(pre, curr_threshold)

                # Skip morphing so in the next cycle the binarization will be corrected before turning it off.
                print(f'{[i]} --Binarize      : skipping attempt {cnt}!')
                continue

        # Pixel-ratio checking:
        black_pixels = imutil.pixel_percentage(p, imutil.BLACK_COLOR)
        white_pixels = imutil.pixel_percentage(p, imutil.WHITE_COLOR)
        if black_pixels > max_pixel_limit or white_pixels > max_pixel_limit:
            print(
                f'{[i]} --Pixel-check   : [black={black_pixels:,.2f}%, white={white_pixels:,.2f}%] Exceeded {max_pixel_limit:}% limit!')
            break
        else:
            print(
                f'{[i]} --Pixel-check   : [black={black_pixels:,.2f}%, white={white_pixels:,.2f}%] within {max_pixel_limit:}% limit.')

        print(f'[{i}] --Morphing      : dilation:erosion=[{kwargs["dilate_iteration"]}:{kwargs["erode_iteration"]}]')

        p = cvlib.morph_proportionate_close(p, kwargs["dilate_iteration"], kwargs["erode_iteration"],
                                            kwargs['dilate_size'], kwargs['erode_size'])

        contour = shape.find_rectangle(processed_img=p,
                                       min_area_factor=kwargs['min_area_factor'],
                                       cnt=cnt,
                                       box=kwargs['box'],
                                       draw=False,
                                       verbose=False)
        if contour is not None:
            cropped = imutil.crop_roi(image, contour, imutil.GREEN_COLOR)
            break

    print(f'----------> [END, attempt={cnt}/{kwargs["attempt_limit"]}]\n')
    return cropped, p


def detect_barcode_v4(image, **kwargs):
    """
    Processes the given image to detect barcode using the parameters documented in the module section.
    :return: The detected rectangle's coordinates.
    """
    max_pixel_limit = int(kwargs['max_pixel_limit'])
    box = None
    cnt = 0
    # Assign current-threshold for processing with config-threshold.
    curr_threshold = int(kwargs['min_threshold'])

    pre = preprocess_image2(image, kwargs['gamma'], kwargs['gaussian_ksize'], kwargs['gaussian_sigma'])
    image_size = image.shape[0] * image.shape[1]
    print(f'Info                : size={image_size:,}, RxC=[{pre.shape[0]:,}x{pre.shape[1]:,}], '
          f'box-ratio-on: {kwargs["box"]}, attempt-limit: {kwargs["attempt_limit"]}, min-threshold={curr_threshold}')

    p = cvlib.average_blur(pre, (9, 9))

    p = cvlib.detect_gradient(p)

    p = cvlib.average_blur(p, (3, 3))

    p = cvlib.binarize(p, curr_threshold)

    for i in range(1, int(kwargs['attempt_limit']) + 1):
        cnt = i
        print(f'----------> [BEGIN, attempt={cnt}]')

        # Pixel-ratio checking:
        black_pixels = imutil.pixel_percentage(p, imutil.BLACK_COLOR)
        white_pixels = imutil.pixel_percentage(p, imutil.WHITE_COLOR)
        if black_pixels > max_pixel_limit or white_pixels > max_pixel_limit:
            print(
                f'{[i]} --Pixel-check   : [black={black_pixels:,.2f}%, white={white_pixels:,.2f}%] Exceeded {max_pixel_limit:}% limit!')
            break
        else:
            print(
                f'{[i]} --Pixel-check   : [black={black_pixels:,.2f}%, white={white_pixels:,.2f}%] within {max_pixel_limit:}% limit.')

        print(f'[{i}] --Morphing      : dilation:erosion=[{kwargs["dilate_iteration"]}:{kwargs["erode_iteration"]}]')

        p = cvlib.morph_proportionate_close(p, kwargs["dilate_iteration"], kwargs["erode_iteration"],
                                            kwargs['dilate_size'], kwargs['erode_size'])

        vertical_se = np.ones((2, 1)).astype('uint8')
        p = cvlib.morph_erode_ex(p, vertical_se, 2)

        # p = cvlib.morph_proportionate_close(p, 2,1, 3, 2)
        p = cvlib.morph_proportionate_close(p, 2, 1, 2, 1)

        # TODO: To reconsider if needed because few parameters are not compatible.
        # p = cvlib.morph_open(p, (45, 45))
        # p = cvlib.morph_open(p, (15, 15))

        # p = cvlib.gaussian_blur(p, kwargs['gaussian_ksize'], kwargs['gaussian_sigma'])

        contour = shape.find_rectangle(processed_img=p,
                                       min_area_factor=kwargs['min_area_factor'],
                                       cnt=cnt,
                                       box=kwargs['box'],
                                       draw=True,
                                       verbose=False)
        if contour is not None:
            rect = cv2.minAreaRect(contour)
            box = np.intp(cv2.boxPoints(rect))
            # TODO: Is this break stopping anything larger? This makes more than one ROI. So, need to choose the larger??
            break

    print(f'----------> [END, attempt={cnt}/{kwargs["attempt_limit"]}]\n')
    return box, p


def detect_barcode_v3(image, **kwargs):
    """
    Processes the given image to detect barcode using the parameters documented in the module section.
    :return: The detected rectangle's coordinates.
    """
    max_pixel_limit = int(kwargs['max_pixel_limit'])
    box = None
    cnt = 0
    # Assign current-threshold for processing with config-threshold.
    curr_threshold = int(kwargs['min_threshold'])

    pre = preprocess_image(image, kwargs['gamma'], kwargs['gaussian_ksize'], kwargs['gaussian_sigma'])
    image_size = image.shape[0] * image.shape[1]
    print(f'Info                : size={image_size:,}, RxC=[{pre.shape[0]:,}x{pre.shape[1]:,}], '
          f'box-ratio-on: {kwargs["box"]}, attempt-limit: {kwargs["attempt_limit"]}')

    p = cvlib.binarize_inv(pre, curr_threshold)

    for i in range(1, int(kwargs['attempt_limit']) + 1):
        cnt = i
        print(f'----------> [BEGIN, attempt={cnt}]')

        # Pixel-ratio checking:
        black_pixels = imutil.pixel_percentage(p, imutil.BLACK_COLOR)
        white_pixels = imutil.pixel_percentage(p, imutil.WHITE_COLOR)
        if black_pixels > max_pixel_limit or white_pixels > max_pixel_limit:
            print(
                f'{[i]} --Pixel-check   : [black={black_pixels:,.2f}%, white={white_pixels:,.2f}%] Exceeded {max_pixel_limit:}% limit!')
            break
        else:
            print(
                f'{[i]} --Pixel-check   : [black={black_pixels:,.2f}%, white={white_pixels:,.2f}%] within {max_pixel_limit:}% limit.')

        print(f'[{i}] --Morphing      : dilation:erosion=[{kwargs["dilate_iteration"]}:{kwargs["erode_iteration"]}]')

        p = cvlib.morph_proportionate_close(p, kwargs["dilate_iteration"], kwargs["erode_iteration"],
                                            kwargs['dilate_size'], kwargs['erode_size'])

        # TODO: To reconsider if needed because few parameters are not compatible.
        # p = cvlib.morph_open(p, (45, 45))
        # p = cvlib.morph_open(p, (15, 15))

        p = cvlib.gaussian_blur(p, kwargs['gaussian_ksize'], kwargs['gaussian_sigma'])

        contour = shape.find_rectangle(processed_img=p,
                                       min_area_factor=kwargs['min_area_factor'],
                                       cnt=cnt,
                                       box=kwargs['box'],
                                       draw=True,
                                       verbose=False)
        if contour is not None:
            rect = cv2.minAreaRect(contour)
            box = np.intp(cv2.boxPoints(rect))
            # TODO: Is this break stopping anything larger? This makes more than one ROI. So, need to choose the larger??
            break

    print(f'----------> [END, attempt={cnt}/{kwargs["attempt_limit"]}]\n')
    return box, p


def detect_qrcode(image, **kwargs):
    """
    Processes the given image to detect qrcode using the parameters documented in the module section.
    :param image: The image to process.
    :return: The detected rectangle's coordinates.
    """
    pre = preprocess_image(image, kwargs['gamma'], kwargs['gaussian_ksize'], kwargs['gaussian_sigma'])
    binary = cv2.threshold(pre, kwargs['thresh_min'], 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    vertical = cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 3))
    horizontal = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 1))
    p = binary
    box = None
    for i in range(1, 3):
        # Dilate both ways in same ratio
        p = cvlib.morph_dilate_ex(p, vertical, 3)
        p = cvlib.morph_dilate_ex(p, horizontal, 3)

        # Erode both ways in same ratio
        p = cvlib.morph_erode_ex(p, horizontal, 2)
        p = cvlib.morph_erode_ex(p, vertical, 2)

        # Close the openings.
        p = cvlib.morph_close(p, (3, 3))

        # Dilate both ways in same ratio
        p = cvlib.morph_dilate_ex(p, vertical, 1)
        p = cvlib.morph_dilate_ex(p, horizontal, 1)

        # Close the openings.
        p = cvlib.morph_close(p, (3, 3))

        contour = shape.find_rectangle(processed_img=p,
                                       min_area_factor=kwargs['min_area_factor'],
                                       cnt=i,
                                       box=kwargs['box'],
                                       draw=False,
                                       verbose=False)
        if contour is not None:
            rect = cv2.minAreaRect(contour)
            box = np.intp(cv2.boxPoints(rect))
            break
    return box


def decode_barcode(img):
    """
    This function is used only to verify the detected barcode.
    :param img: The cropped image with barcode detected.
    :return: barcode when detected.
    """
    try:
        detector = cv2.barcode_BarcodeDetector()
        # decoded_text, points, barcode_type = detector.detectAndDecode(img)
        decoded_text, _, _ = detector.detectAndDecode(img)
        return decoded_text
    except Exception as e:
        print(f"scanner: [{threading.currentThread().native_id}] Error: {e}")
        return None


def decode_qrcode(img):
    """
    This function is used only to verify the detected qrcode.
    :param img: The cropped image with barcode detected.
    :return: qrcode when detected.
    """
    try:
        detector = cv2.QRCodeDetector()
        decoded_text, _, _ = detector.detectAndDecode(img)
        return decoded_text
    except Exception as e:
        print(f"scanner: [{threading.currentThread().native_id}] Error: {e}")
        return None
