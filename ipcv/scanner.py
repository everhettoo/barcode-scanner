import cv2

from ipcv import cvlib


def preprocess(image, gamma, gaussian_ksize, gaussian_sigma):
    # Adjust the contrast
    p = cvlib.adjust_gamma(image, gamma)

    # Convert image to gray scale for processing.
    p = cvlib.convert_rgb2gray(p)

    # Remove noise.
    p = cvlib.gaussian_blur(p, gaussian_ksize, gaussian_sigma)

    return p


def detect_qrcode(**kwargs):
    p = kwargs['image'].copy()
    p = cvlib.binarize(p, kwargs['thresh_min'])

    ksize = (13,13)
    iteration = 5
    p = cvlib.morph_close(p, ksize)
    p = cvlib.morph_open(p, ksize)
    p = cvlib.morph_dilate(p, iteration, ksize)
    p = cvlib.morph_erode(p, iteration)

    contours, _ = cv2.findContours(p, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



def detect_barcode(**kwargs):
    """
    Processed the given image to detect barcode using the following parameters:.
    :param kwargs:
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
    :return: The detected barcodes are annotated on the original image first, and cropped barcode is returned.
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

    cropped = cvlib.get_prominent_contour(kwargs['image'], p, kwargs['offset'])

    return cropped


def decode_barcode(img):
    """
    This function is used only to verify the detected barcode
    :param img: The cropped image with barcode detected.
    :return:
    """
    detector = cv2.barcode_BarcodeDetector()
    # decoded_text, points, barcode_type = detector.detectAndDecode(img)
    decoded_text, _, _ = detector.detectAndDecode(img)
    if decoded_text == '':
        return None

    return decoded_text
