# -*- coding: utf-8 -*-
from css_utils import gaussian_kernel, compute_curvature
import cv2
import numpy as np
import math

class CurvatureScaleSpace(object):
    """ Curvature Scale Space
    A simple curvature scale space implementation based on
    Mohkatarian et. al. paper. Full algorithm detailed in
    Okal msc thesis
    """
    
    def get_points_from_img(self, image, simpleto=100):
        """
            This is much faster version of getting shape points algo.
            It's based on cv2.findContours algorithm, which is basically return shape points
            ordered by curve direction. So it's gives better and faster result
        """
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        cnts = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        max_cnt = []
        for contour in cnts[1]:
            if len(contour) > len(max_cnt):
                max_cnt = contour
        points = np.array(cnts[1][0]).reshape((-1, 2))
        if len(cnts[1]) > 1:
            points = np.concatenate([points, np.array(cnts[1][1]).reshape((-1, 2))], axis=0)
        points = points.tolist()
        step = math.floor(len(points) / simpleto)
        step = math.floor(len(points) / simpleto)
        if step is 0:
            step = 1
        points = [points[i] for i in range(0, len(points), step)][:simpleto]
        if len(points) < simpleto:
            points = points + [[0, 0]] * (simpleto - len(points))
        return np.transpose(points)

    def __init__(self):
        pass

    def find_zero_crossings(self, kappa):
        """ find_zero_crossings(kappa)
        Locate the zero crossing points of the curvature signal kappa(t)
        """

        crossings = []

        for i in range(0, kappa.size - 2):
            if (kappa[i] < 0.0 and kappa[i + 1] > 0.0) or (kappa[i] > 0.0 and kappa[i + 1] < 0.0):
                crossings.append(i)

        return crossings

    def generate_css(self, curve, max_sigma, step_sigma):
        """ generate_css(curve, max_sigma, step_sigma)
        Generates a CSS image representation by repetatively smoothing the initial curve L_0 with increasing sigma
        """

        cols = curve[0, :].size
        rows = math.floor(max_sigma / step_sigma)
        css = np.zeros(shape=(rows, cols))

        srange = np.linspace(1, max_sigma - 1, rows)
        for i, sigma in enumerate(srange):
            kappa, sx, sy = compute_curvature(curve, sigma)

            # find interest points
            xs = self.find_zero_crossings(kappa)

            # save the interest points
            if len(xs) > 0 and sigma < max_sigma - 1:
                for c in xs:
                    css[i, c] = sigma  # change to any positive

            else:
                return css

    def generate_visual_css(self, rawcss, closeness, return_all=False):
        """ generate_visual_css(rawcss, closeness)
        Generate a 1D signal that can be plotted to depict the CSS by taking
        column maximums. Further checks for close interest points and nicely
        smoothes them with weighted moving average
        """

        flat_signal = np.amax(rawcss, axis=0)

        # minor smoothing via moving averages
        window = closeness
        weights = gaussian_kernel(window, 0, window, False)  # gaussian weights
        sig = np.convolve(flat_signal, weights)[window - 1:-(window - 1)]

        maxs = []

        # get maximas
        w = sig.size

        for i in range(1, w - 1):
            if sig[i - 1] < sig[i] and sig[i] > sig[i + 1]:
                maxs.append([i, sig[i]])

        if return_all:
            return sig, maxs
        else:
            return sig

    def generate_eigen_css(self, rawcss, return_all=False):
        """ generate_eigen_css(rawcss, return_all)
        Generates Eigen-CSS features
        """
        rowsum = np.sum(rawcss, axis=0)
        csum = np.sum(rawcss, axis=1)

        # hack to trim c
        colsum = csum[0:rowsum.size]

        freq = np.fft.fft(rowsum)
        mag = abs(freq)

        tilde_rowsum = np.fft.ifft(mag)

        feature = np.concatenate([tilde_rowsum, colsum], axis=0)

        if not return_all:
            return feature
        else:
            return feature, rowsum, tilde_rowsum, colsum
