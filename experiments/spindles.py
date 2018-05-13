from data.loaders import load_alpha_delay_subjects
from models import SWAnalyticFilter2, RLS
import pylab as plt
from helpers import band_hilbert, get_ideal_H, cLS, get_XY
import numpy as np
from scipy.signal import firwin, lfilter, filtfilt, stft


x = load_alpha_delay_subjects()

fs = 500
band = np.array((8, 12))
x = x[:fs*60*10]*1000000


n_fft = 2000
n_taps = 1000
F = np.array([np.exp(-2j*np.pi/n_fft*k*np.arange(n_taps)) for k in np.arange(n_fft)])

fig, axes = plt.subplots(3, 2, sharex=True, sharey=True)

for j, d in enumerate([200, 0, 100, -50, 50, 'rand']):
    if d != 'rand':
        #if d<999:
        #plt.close()
        H = get_ideal_H(n_fft, fs, band, d)
        b_cfir_f = cLS(F, H, 0)

        X_train, Y_train = get_XY(x[:fs*60*2], band_hilbert(x[:fs*60*2], fs, (8, 12)), n_taps, d)
        b_cfir_f = cLS(X_train, Y_train, 0)
        if d>0:
            b_fir_band = firwin(d, band / fs * 2, pass_zero=False)
            b_fir_smooth = firwin(d, 2 / fs * 2)
            rec_fir = lfilter(b_fir_smooth, [1], np.abs(lfilter(b_fir_band, [1], x)))
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
    start = int(fs*0.7)
    stop = int(fs*0.5)
    t = np.arange(-start, stop)/fs*1000

    events = np.where(rec>th)[0]
    #events = df['signal_P4'].where(df['signal_P4']>df['signal_Composite']).dropna().index
    for k in events:
        if k-k0>500 and k>start and k<len(x)-stop:
            real_trials.append(env[k-start:k+stop])
            k0 = k

    real = np.array(real_trials)
    axes[j//2, j%2].plot(t, real.mean(0), 'k', label='T-cFIR')
    axes[j//2, j%2].fill_between(t, real.mean(0)-real.std(0), real.mean(0)+real.std(0), alpha=0.2, color='k')
    axes[j//2, j%2].axvline(0, color='k', alpha=0.8)



    if d!='rand' and d>0:
        real_trials = []
        k0 = 0
        th = np.percentile(rec_fir, 95)
        events = np.where(rec_fir > th)[0]
        # events = df['signal_P4'].where(df['signal_P4']>df['signal_Composite']).dropna().index
        for k in events:
            if k - k0 > 500 and k > start and k < len(x) - stop:
                real_trials.append(env[k - start:k + stop])
                k0 = k

        real = np.array(real_trials)
        axes[j//2, j%2].plot(t, real.mean(0), 'r--', label='FIR+Abs')
        axes[j//2, j%2].fill_between(t, real.mean(0) - real.std(0), real.mean(0) + real.std(0), alpha=0.2, color='r')
        if d == 200:
            axes[j // 2, j % 2].legend()

    axes[j // 2, j % 2].text(-690, 0.55, 'd = {}ms'.format(d) if d!='rand' else 'random')

axes[2, 0].set_xlabel('Time, ms')
axes[2, 1].set_xlabel('Time, ms')
axes[1, 0].set_ylabel('Envelope, $\mu$V')

fig.subplots_adjust(hspace=0)
fig.subplots_adjust(wspace=0)

plt.savefig('spindles.png', dpi=200)
plt.show()