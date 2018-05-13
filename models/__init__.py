import numpy as np
from scipy import fftpack


class SWAnalyticFilter2:
    def __init__(self, fs, band, n_samples, n_fft=None, n_channels=1, chank_ratio=None):
        self.n_fft = n_fft or n_samples
        self.n_samples = n_samples
        self.center = self.n_samples // 2
        w = fftpack.fftfreq(self.n_fft, d=1. / fs)
        self.w_mask = (w < band[0]) | (w > band[1])
        self.buffer = np.zeros((n_samples, n_channels) if n_channels > 1 else n_samples)
        self.chank_ratio = chank_ratio or n_samples

    def apply(self, x):
        n_x = x.shape[0]
        if n_x <= self.n_samples // self.chank_ratio:
            self.buffer = np.roll(self.buffer, -n_x)
            self.buffer[-n_x:] = x
            Xf = fftpack.fft(self.buffer, self.n_fft, axis=0)
            Xf[self.w_mask] = 0
            res = fftpack.ifft(Xf, axis=0)[:self.n_samples]
            return 2 * res[-self.center - n_x:-self.center]
        else:
            return np.concatenate([self.apply(x[j:j + self.n_samples // self.chank_ratio]) for j in
                                   range(0, n_x, self.n_samples // self.chank_ratio)])


class RLS:
    def __init__(self, p, lam, delta):
        self.w = np.zeros(p, dtype='complex128')
        self.P = delta * np.eye(p)
        self.lam = lam

    def adapt(self, x, d):
        alpha = d - np.dot(x, self.w)
        _lam = np.dot(x, np.dot(self.P, x))
        g = np.dot(self.P, x.conj()) / (self.lam + _lam)
        self.P = self.P - np.dot(g[:, None], np.dot(x[None, :], self.P))
        self.P /= self.lam
        self.w += alpha * g  # - 0.01*self.w

    def predict(self, x):
        return np.dot(x, self.w)