import numpy as np

class BaseFun:
    def __init__(self) -> None:
        pass

    def _psnr(self, img1, img2, max_pixel=1):
        img1, img2 = np.array(img1, dtype='float'), np.array(img2, dtype='float')
        mse = np.mean((img1 - img2) ** 2)
        psnr = 10 * np.log10(max_pixel * max_pixel / mse)
        return psnr

    # SSIM
    def _ssim(self, img1, img2, K=(0.01, 0.03), L=1):
        C1, C2 = (K[0] * L) ** 2, (K[1] * L) ** 2
        C3 = C2 / 2
        img1, img2 = np.array(img1, dtype='float'), np.array(img2, dtype='float')
        m1, m2 = np.mean(img1), np.mean(img2)
        s1, s2 = np.std(img1), np.std(img2)
        s12 = np.mean((img1 - m1) * (img2 - m2))
        l = (2 * m1 * m2 + C1) / (m1**2 + m2**2 + C1)
        c = (2 * s1 * s2 + C2) / (s1**2 + s2**2 + C2)
        s = (s12 + C3) / (s1 * s2 + C3)
        return l * c * s