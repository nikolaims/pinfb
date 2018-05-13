from data.loaders import load_alpha_delay_subjects
from models import SWAnalyticFilter2, RLS
import pylab as plt
from helpers import band_hilbert, get_ideal_H, cLS, get_XY, band_hilbert2
import numpy as np
from scipy.signal import firwin, lfilter, filtfilt, stft, minimum_phase
import pickle

fig = plt.figure(figsize=(8, 3))
x = load_alpha_delay_subjects()
#x = np.random.normal(size=x.shape)
fs = 500
band = np.array((8, 12))

n_fft = 2000
n_taps = 500
F = np.array([np.exp(-2j*np.pi/n_fft*k*np.arange(n_taps)) for k in np.arange(n_fft)])

delays = np.arange(1, 2, 10)

indx, step = np.linspace(0, len(x), 10, retstep=True, endpoint=False, dtype=int)
indx = indx[:6]

corrs_fir = np.zeros((len(indx)-1, len(delays)))*np.nan
corrs_cfir_f = np.zeros((len(indx) - 1, len(delays)))*np.nan
corrs_cfir_wf = np.zeros((len(indx) - 1, len(delays)))*np.nan
corrs_cfir_t = np.zeros((len(indx) - 1, len(delays)))*np.nan
corrs_hilbert = np.zeros((len(indx) - 1, len(delays)))*np.nan
corrs_rls = np.zeros((len(indx) - 1, len(delays)))*np.nan
min_phase_delays = np.zeros((len(indx) - 1, len(delays)))*np.nan
corrs_min_phase = np.zeros((len(indx) - 1, len(delays)))*np.nan


d = 0
for k, kk in enumerate(indx[2:]):
    print('****', k)
    x_train = x[indx[k-1]:indx[k-1]+int(step)]
    x_std = x_train.std()
    x_mean = x_train.mean()
    x_train = (x_train - x_mean)/x_std
    y_train = band_hilbert(x_train, 500, band)
    x_test = x[kk:kk+int(step)]
    y_test = band_hilbert(x_test, 500, band)


    X_train, Y_train = get_XY(x_train, y_train, n_taps, d)
    b_cfir_t = cLS(X_train, Y_train, 0)
    rec_cfir_t = np.abs(lfilter(b_cfir_t, [1], x_test))[d:]

    X_train, Y_train = get_XY(x_train, y_train, n_taps, 50)
    b_cfir_t = cLS(X_train, Y_train, 0)
    rec_cfir_t50 = np.abs(lfilter(b_cfir_t, [1], x_test))

    X_train, Y_train = get_XY(x_train, y_train, n_taps, 100)
    b_cfir_t = cLS(X_train, Y_train, 0)
    rec_cfir_t100 = np.abs(lfilter(b_cfir_t, [1], x_test))


    tar = np.abs(y_test)[:-d if d>0 else None]

    plt.plot(np.arange(len(rec_cfir_t100)) / fs, rec_cfir_t100/rec_cfir_t100.std(), '--', color='k', label='(c) 200 ms')

    plt.plot(np.arange(len(rec_cfir_t50)) / fs, rec_cfir_t50/rec_cfir_t50.std(), color='k', label='(b) 100 ms')


    plt.plot(np.arange(len(rec_cfir_t))/fs, rec_cfir_t/rec_cfir_t.std(), color='#01adf5', label='(a) 0 ms')
    plt.plot(np.arange(len(tar))/fs, tar/tar.std(), color='r', label='Ground truth')
    plt.xlabel('Time, s')
    plt.ylabel('Envelope')
    plt.xlim(10, 17)
    plt.ylim(0, 7)
    plt.legend()
    plt.title('C. T-cFIR envelope reconstruction example')
    plt.tight_layout()
    plt.savefig('env_rec.png', dpi=200)
    plt.show()

