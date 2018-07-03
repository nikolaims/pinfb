import numpy as np
import pylab as plt
from scipy.signal import stft, lfilter
from scipy import fftpack
import pandas as pd



def cLS(X, Y, lambda_=0):
    reg = lambda_*np.eye(X.shape[1])
    b = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X.conj())+reg), X.T.conj()), Y)
    return b

def ridge(X, Y, lambda_=0):
    reg = lambda_ * np.eye(X.shape[1])
    b = np.linalg.solve(X.T.dot(X) + reg, (X.T).dot(Y))
    return b

def band_hilbert(x, fs, band, N=None, axis=-1):
    x = np.asarray(x)
    Xf = fftpack.fft(x, N, axis=axis)
    w = fftpack.fftfreq(x.shape[0], d=1. / fs)
    Xf[(w < band[0]) | (w > band[1])] = 0
    x = fftpack.ifft(Xf, axis=axis)
    return 2*x

def band_hilbert2(x, fs, band, N, axis=-1):
    x = np.asarray(x)
    lenx = x.shape[0]
    Xf = fftpack.fft(x, N, axis=axis)
    w = fftpack.fftfreq(N, d=1. / fs)
    Xf[(w < band[0]) | (w > band[1])] = 0
    x = fftpack.ifft(Xf, axis=axis)
    return 2*x[:lenx]

def get_ideal_H(n_fft, fs, band, delay=0):
    w = np.arange(n_fft)
    H = 2*np.exp(-2j*np.pi*w/n_fft*delay)
    H[(w/n_fft*fs<band[0]) | (w/n_fft*fs>band[1])] = 0
    return H

def get_XY(x, y, n_taps, delay):
    X = np.array([x[n:n+n_taps][::-1] for n in range(len(x) - n_taps - max(0, -delay))])
    Y = np.array([y[n+n_taps-delay] for n in range(len(x) - n_taps - max(0, -delay))])
    return X, Y