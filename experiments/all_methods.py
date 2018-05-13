from data.loaders import load_alpha_delay_subjects
from models import SWAnalyticFilter2, RLS
import pylab as plt
from helpers import band_hilbert, get_ideal_H, cLS, get_XY, band_hilbert2
import numpy as np
from scipy.signal import firwin, lfilter, filtfilt, stft, minimum_phase
import pickle


x = load_alpha_delay_subjects()
x = np.random.normal(size=x.shape)
fs = 500
band = np.array((8, 12))

n_fft = 2000
n_taps = 500
F = np.array([np.exp(-2j*np.pi/n_fft*k*np.arange(n_taps)) for k in np.arange(n_fft)])

delays = np.arange(-51, 250, 10)

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

for k, kk in enumerate(indx[1:]):
    print('****', k)
    x_train = x[indx[k-1]:indx[k-1]+int(step)]
    x_std = x_train.std()
    x_mean = x_train.mean()
    x_train = (x_train - x_mean)/x_std
    y_train = band_hilbert(x_train, 500, band)
    x_test = x[kk:kk+int(step)]/x_std
    x_test = (x_test - x_mean)/x_std
    y_test = band_hilbert(x_test, 500, band)

    freq, time, spec = stft(x_train, fs, nperseg=n_fft, return_onesided=False, noverlap=int(n_fft * 0.9))
    mask = np.abs(spec).mean(1)

    w = np.diag(mask)


    for j, d in enumerate(delays):
        print('delay', d)

        # freq. domain ideal filter
        H = get_ideal_H(n_fft, fs, band, d)
        b_cfir_f = cLS(F, H, 0)
        b_cfir_wf = cLS(np.dot(w, F), np.dot(w, H), 0)

        X_train, Y_train = get_XY(x_train, y_train, n_taps, d)
        b_cfir_t = cLS(X_train, Y_train, 0)

        if d >= 0:
            tar = np.abs(y_test)[:-d]

            b_fir_band = firwin(d, band / fs * 2, pass_zero=False)
            b_fir_smooth = firwin(d, 2 / fs * 2)
            rec_fir = lfilter(b_fir_smooth, [1], np.abs(lfilter(b_fir_band, [1], x_test)))[d:]
            rec_cfir_f = np.abs(lfilter(b_cfir_f, [1], x_test))[d:]
            rec_cfir_wf = np.abs(lfilter(b_cfir_wf, [1], x_test))[d:]

            rec_cfir_t = np.abs(lfilter(b_cfir_t, [1], x_test))[d:]

            corrs_fir[k, j] = np.corrcoef(rec_fir, tar)[0, 1]

            saf = SWAnalyticFilter2(fs, band, d * 2, n_fft)
            chunked = np.zeros((len(x_test), d*2))
            for l in range(d*2, len(x_test)):
                chunked[l, :] = x_test[l-2*d:l]

            rec_hilbert = np.abs(saf.apply(x_test))
            #rec = np.abs(saf.apply(x_test))[d:]
            corrs_hilbert[k, j] = np.corrcoef(rec_hilbert[d:], tar)[0, 1]
            if d>0:
                tar1 = np.abs(y_test)
                b_fir_band = minimum_phase(firwin(d, band / fs * 2, pass_zero=False))
                b_fir_smooth = minimum_phase(firwin(d, 2 / fs * 2))
                rec_fir = lfilter(b_fir_smooth, [1], np.abs(lfilter(b_fir_band, [1], x_test)))
                corr_delays = np.arange(1, 300)
                cross_corr = np.array([np.corrcoef(tar1[:-d1], rec_fir[d1:])[0][1] for d1 in corr_delays])
                opt_delay = corr_delays[np.argmax(cross_corr)]
                min_phase_delays[k, j] = opt_delay
                corrs_min_phase[k, j] = np.max(cross_corr)



        else:
            tar = np.abs(y_test)[-d:]
            rec_cfir_f = np.abs(lfilter(b_cfir_f, [1], x_test))[:d]
            rec_cfir_wf = np.abs(lfilter(b_cfir_wf, [1], x_test))[:d]
            rec_cfir_t = np.abs(lfilter(b_cfir_t, [1], x_test))[:d]

        corrs_cfir_f[k, j] = np.corrcoef(rec_cfir_f, tar)[0, 1]
        corrs_cfir_wf[k, j] = np.corrcoef(rec_cfir_wf, tar)[0, 1]
        corrs_cfir_t[k, j] = np.corrcoef(rec_cfir_t, tar)[0, 1]


plt.plot(delays/fs*1000, corrs_fir.mean(0))
plt.fill_between(delays/fs*1000, corrs_fir.mean(0) - corrs_fir.std(0), corrs_fir.mean(0) + corrs_fir.std(0), alpha=0.3)

plt.plot(delays / fs * 1000, corrs_cfir_f.mean(0))
plt.fill_between(delays / fs * 1000, corrs_cfir_f.mean(0) - corrs_cfir_f.std(0), corrs_cfir_f.mean(0) + corrs_cfir_f.std(0), alpha=0.3)

plt.plot(delays / fs * 1000, corrs_cfir_wf.mean(0))
plt.fill_between(delays / fs * 1000, corrs_cfir_wf.mean(0) - corrs_cfir_wf.std(0), corrs_cfir_wf.mean(0) + corrs_cfir_wf.std(0), alpha=0.3)

plt.plot(delays / fs * 1000, corrs_cfir_t.mean(0))
plt.fill_between(delays / fs * 1000, corrs_cfir_t.mean(0) - corrs_cfir_t.std(0), corrs_cfir_t.mean(0) + corrs_cfir_t.std(0), alpha=0.3)

plt.plot(delays / fs * 1000, corrs_hilbert.mean(0))
plt.fill_between(delays / fs * 1000, corrs_hilbert.mean(0) - corrs_hilbert.std(0), corrs_hilbert.mean(0) + corrs_hilbert.std(0), alpha=0.3)



plt.plot(min_phase_delays.mean(0) / fs * 1000, corrs_min_phase.mean(0))
#plt.fill_between(min_phase_delays.mean(0) / fs * 1000, corrs_rls.mean(0) - corrs_rls.std(0), corrs_rls.mean(0) + corrs_rls.std(0), alpha=0.3)
#plt.plot(delays/fs*1000, corrs_fir.mean(0) - corrs_fir.std(0))
plt.xlabel('Delay, ms')
plt.ylabel('Correlation')
plt.axvline(0, color='k')

with open('curves_8_12Hz_rand.pkl', 'wb') as f:
    pickle.dump({'fir': corrs_fir, 'cfir_f': corrs_cfir_f, 'cfir_wf': corrs_cfir_wf, 'cfir_t': corrs_cfir_t,
                 'hilbert': corrs_hilbert, 'delays': delays, 'min_phase_delays': min_phase_delays,
                 'corrs_min_phase': corrs_min_phase}, f)

plt.legend(['FIR+Abs+Smooth', 'cFIR(fLS)+Abs', 'cFIR(fWLS)', 'cFIR(tLS)', 'wFTT(Hilbert)', 'Min-Phase'])
plt.show()

