from ipcv import cvlib


class Scanner:
    def __init__(self):
        pass

    def detect_barcode(self, image, gamma, gaussian_ksize, gaussian_sigma, avg_ksize1, avg_ksize2, thresh_min,
                       dilate_kernel, dilate_iteration, shrink_factor):
        """
        Processed the given image and detect barcodes.
        :param image: The processed image.
        :param gamma: The gamma value to use.
        :param gaussian_ksize: The gaussian_ksize to use.
        :param gaussian_sigma: The gaussian_sigma to use.
        :param avg_ksize1: The avg_ksize to use.
        :param avg_ksize2:  The avg_ksize to use (only [3,3] works for now).
        :param thresh_min: The thresh_min.
        :param dilate_kernel: The dilate_kernel to use.
        :param dilate_iteration: The dilate_iteration.
        :param shrink_factor:  The shrink_factor to use.
        :return: The detected barcodes drawn image and cropped barcode.
        """
        p = cvlib.convert_rgb2gray(image)

        p = cvlib.adjust_gamma(p, gamma)

        p = cvlib.gaussian_blur(p, gaussian_ksize, gaussian_sigma)

        p = cvlib.average_blur(p, avg_ksize1)

        p = cvlib.detect_gradient(p)

        # Only [3,3] works
        p = cvlib.average_blur(p, avg_ksize2)

        p = cvlib.binarize(p, thresh_min)

        p = cvlib.dilate(p, dilate_kernel, dilate_iteration)

        # Shrink
        x = p
        new_width = int(p.shape[1] / shrink_factor)
        new_height = int(p.shape[0] / shrink_factor)

        p = cvlib.resize_image(p, new_width, new_height)

        p = cvlib.resize_image(p, x.shape[1], x.shape[0])

        cropped = cvlib.get_prominent_contour(image, p)

        return cropped

    def detect_barcode5(self, **kwargs):
        """
        Processed the given image and detect barcodes.
        :param image: The processed image.
        :param gamma: The gamma value to use.
        :param gaussian_ksize: The gaussian_ksize to use.
        :param gaussian_sigma: The gaussian_sigma to use.
        :param avg_ksize1: The avg_ksize to use.
        :param avg_ksize2:  The avg_ksize to use (only [3,3] works for now).
        :param thresh_min: The thresh_min.
        :param dilate_kernel: The dilate_kernel to use.
        :param dilate_iteration: The dilate_iteration.
        :param shrink_factor:  The shrink_factor to use.
        :return: The detected barcodes drawn image and cropped barcode.
        """
        p = cvlib.convert_rgb2gray(kwargs['image'])

        p = cvlib.adjust_gamma(p, kwargs['gamma'])

        p = cvlib.gaussian_blur(p, kwargs['gaussian_ksize'], kwargs['gaussian_sigma'])

        p = cvlib.average_blur(p, kwargs['avg_ksize1'])

        p = cvlib.detect_gradient(p)

        # Only [3,3] works
        p = cvlib.average_blur(p, kwargs['avg_ksize2'])

        p = cvlib.binarize(p, kwargs['thresh_min'])

        p = cvlib.dilate(p, kwargs['dilate_kernel'], kwargs['dilate_iteration'])

        # Shrink
        x = p
        new_width = int(p.shape[1] / kwargs['shrink_factor'])
        new_height = int(p.shape[0] / kwargs['shrink_factor'])

        p = cvlib.resize_image(p, new_width, new_height)

        p = cvlib.resize_image(p, x.shape[1], x.shape[0])

        cropped = cvlib.get_prominent_contour(kwargs['image'], p)

        return cropped
