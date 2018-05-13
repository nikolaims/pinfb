from data.loaders import load_alpha_delay_subjects
from models import SWAnalyticFilter2, RLS
import pylab as plt
from helpers import band_hilbert, get_ideal_H, cLS, get_XY
import numpy as np
from scipy.signal import firwin, lfilter, filtfilt, stft, minimum_phase


x = load_alpha_delay_subjects()
fs = 500
band = (8, 12)
n_fft = 2000
n_taps = 500
d = 100

x_test = x[:fs*60*2]
y_test = band_hilbert(x_test, 500, band)
tar = np.abs(y_test)

delays = np.arange(10, 300, 10)
min_phase_delays = np.zeros_like(delays)
corrs_min_phase = np.zeros_like(delays)*0.

fir_delays = np.zeros_like(delays)
corrs_fir = np.zeros_like(delays)*0.
for j, d in enumerate(delays):
    print('delay', d)
    b_fir_band = minimum_phase(firwin(d, [8 / fs * 2, 12 / fs * 2], pass_zero=False))
    b_fir_smooth = minimum_phase(firwin(d, 2 / fs * 2))
    rec_fir = lfilter(b_fir_smooth, [1], np.abs(lfilter(b_fir_band, [1], x_test)))
    corr_delays = np.arange(1, 300)
    cross_corr = np.array([np.corrcoef(tar[:-d], rec_fir[d:])[0][1] for d in corr_delays])
    opt_delay = corr_delays[np.argmax(cross_corr)]
    min_phase_delays[j] = opt_delay
    corrs_min_phase[j] = np.max(cross_corr)


    b_fir_band = (firwin(d, [8 / fs * 2, 12 / fs * 2], pass_zero=False))
    b_fir_smooth = (firwin(d, 2 / fs * 2))
    rec_fir = lfilter(b_fir_smooth, [1], np.abs(lfilter(b_fir_band, [1], x_test)))
    corr_delays = np.arange(1, 300)
    cross_corr = np.array([np.corrcoef(tar[:-d], rec_fir[d:])[0][1] for d in corr_delays])
    opt_delay = corr_delays[np.argmax(cross_corr)]
    fir_delays[j] = opt_delay
    corrs_fir[j] = np.max(cross_corr)

    print(opt_delay/fs)
plt.plot(min_phase_delays / fs, corrs_min_phase)
plt.plot(fir_delays / fs, corrs_fir)
print(fir_delays-delays)
print(min_phase_delays-delays)
plt.show()

