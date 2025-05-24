from unittest import TestCase
import numpy as np
from scipy import stats as st
import ipcv.image_processor as ip


class Test(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mat = np.array([[30, 120, 255],
                            [0, 255, 127],
                            [120, 0, 120]],
                           np.uint8)

    def print_result(self, a, b):
        print(
            f'A: Mode = {st.mode(a, axis=None)}, Max = {a.max()}, Min = {a.min()}, Mean = {a.mean():.2f}, SD = {a.std():.2f}')
        print(
            f'B: Mode = {st.mode(b, axis=None)}, Max = {b.max()}, Min = {b.min()}, Mean = {b.mean():.2f}, SD = {b.std():.2f}')
        print(f'A:{a}')
        print(f'B:{b}')

    def test_adjust_gamma(self):
        f = self.mat.astype(np.float32)
        # PEMDAS Rule applied E.g. for value 30 when gamma=0.2: ((30/255)^0.2)*255 = 166.20
        g = ip.adjust_gamma(f, 0.2)
        self.print_result(self.mat, g)

    def test_load_gaussian_blur(self):
        f = self.mat.astype(np.float32)
        g = ip.gaussian_blur(f, (21, 21), 2)
        self.print_result(self.mat, g)

    def test_load_average_blur(self):
        f = self.mat.astype(np.float32)
        g = ip.average_blur(f, (9, 9))
        self.print_result(self.mat, g)
