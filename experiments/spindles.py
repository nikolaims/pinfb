from data.loaders import load_alpha_delay_subjects
from models import SWAnalyticFilter2, RLS
import pylab as plt
from helpers import band_hilbert, get_ideal_H, cLS, get_XY
import numpy as np
from scipy.signal import firwin, lfilter, filtfilt, stft


x = load_alpha_delay_subjects()

fs = 500
band = (8, 12)
x = x[:fs*60*10]*1000000


n_fft = 2000
n_taps = 1000
F = np.array([np.exp(-2j*np.pi/n_fft*k*np.arange(n_taps)) for k in np.arange(n_fft)])

for d in [100, 50, 0, -50, 'rand']:
    if d != 'rand':
        #if d<999:
        #plt.close()
        H = get_ideal_H(n_fft, fs, band, d)
        b_cfir_f = cLS(F, H, 0)
        #plt.plot(b_cfir_f)
        #plt.show()

        #plt.plot(x)

        rec = np.abs(lfilter(b_cfir_f, [1], x))
    else:
        rec = np.random.normal(size=len(x))
    env = np.abs(band_hilbert(x, fs, (8, 12)))
    env = env / env.max()
    rec = rec/rec.max()
    #plt.plot(env)
    #plt.plot(rec)

    th = np.percentile(rec, 95)
    #plt.axhline(th, color='k')
    #plt.show()


    real_trials = []
    k0 = 0
    start = fs*1
    stop = fs*1
    t = np.arange(-start, stop)/fs*1000

    events = np.where(rec>th)[0]
    #events = df['signal_P4'].where(df['signal_P4']>df['signal_Composite']).dropna().index
    for k in events:
        if k-k0>500 and k>start and k<len(x)-stop:
            real_trials.append(env[k-start:k+stop])
            k0 = k

    real = np.array(real_trials)
    plt.plot(t, real.mean(0), label='d={}'.format(d))
    plt.fill_between(t, real.mean(0)-real.std(0), real.mean(0)+real.std(0), alpha=0.5)
    plt.axvline(0, color='k', alpha=0.8)
plt.legend()
plt.xlabel('Time, ms')
plt.ylabel('Envelope, $\mu$V')
plt.show()