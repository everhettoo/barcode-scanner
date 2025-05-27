from unittest import TestCase

from ipcv import cvlib, scanner


class Test(TestCase):
    def test_decode_barcode(self):
        image = cvlib.load_image('../resources/barcode/barcode310.jpg')
        cropped = scanner.detect_barcode(image=image,
                                         gamma=0.5,
                                         gaussian_ksize=(15, 15),
                                         gaussian_sigma=2,
                                         avg_ksize1=(9, 9),
                                         avg_ksize2=(3, 3),
                                         thresh_min=200,
                                         dilate_kernel=(21, 7),
                                         dilate_iteration=4,
                                         shrink_factor=6,
                                         offset=0)
        barcode = scanner.decode_barcode(cropped)
        self.assertIsNotNone(barcode)
        print(barcode)
