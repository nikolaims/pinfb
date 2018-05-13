import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns
#sns.set_palette(sns.color_palette('Paired'))
fs = 500

fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(8, 4))


with open('curves_8_12Hz.pkl', 'rb') as handle:
    b = pickle.load(handle)

corrs_fir = b['fir']
corrs_cfir_f = b['cfir_f']
corrs_cfir_wf = b['cfir_wf']
corrs_cfir_t = b['cfir_t']
corrs_hilbert = b['hilbert']
delays = b['delays']
min_phase_delays = b['min_phase_delays']
corrs_min_phase = b['corrs_min_phase']
axes[0].plot(delays/fs*1000, corrs_fir.mean(0), label='FIR', color='k')
axes[0].fill_between(delays/fs*1000, corrs_fir.mean(0) - corrs_fir.std(0), corrs_fir.mean(0) + corrs_fir.std(0), alpha=0.3, linewidth=0., color='k')

axes[0].plot(min_phase_delays.mean(0) / fs * 1000, corrs_min_phase.mean(0), '--', label='MinPh-FIR', color='k')
axes[0].fill_between(min_phase_delays.mean(0) / fs * 1000, corrs_min_phase.mean(0) - corrs_min_phase.std(0), corrs_min_phase.mean(0) + corrs_min_phase.std(0), alpha=0.3, linewidth=0., color='k')


axes[0].plot(delays / fs * 1000, corrs_hilbert.mean(0), '-.', label='Hilbert-FFT', color='k')
axes[0].fill_between(delays / fs * 1000, corrs_hilbert.mean(0) - corrs_hilbert.std(0), corrs_hilbert.mean(0) + corrs_hilbert.std(0), alpha=0.3, linewidth=0., color='k')


axes[0].plot(delays / fs * 1000, corrs_cfir_f.mean(0), label='F-cFIR', color='r')
axes[0].fill_between(delays / fs * 1000, corrs_cfir_f.mean(0) - corrs_cfir_f.std(0), corrs_cfir_f.mean(0) + corrs_cfir_f.std(0), alpha=0.3, linewidth=0., color='r')

axes[0].plot(delays / fs * 1000, corrs_cfir_wf.mean(0), '--', label='WF-cFIR', color='r')
axes[0].fill_between(delays / fs * 1000, corrs_cfir_wf.mean(0) - corrs_cfir_wf.std(0), corrs_cfir_wf.mean(0) + corrs_cfir_wf.std(0), alpha=0.3, linewidth=0., color='r')

axes[0].plot(delays / fs * 1000, corrs_cfir_t.mean(0), label='T-cFIR', color='#01adf5')
axes[0].fill_between(delays / fs * 1000, corrs_cfir_t.mean(0) - corrs_cfir_t.std(0), corrs_cfir_t.mean(0) + corrs_cfir_t.std(0), alpha=0.3, linewidth=0., color='#01adf5')


axes[0].set_xlabel('Delay, ms')
axes[0].set_ylabel('Correlation')
axes[0].axvline(0, color='k', alpha=0.5)
axes[0].set_xticks(np.arange(-100, 201, step=50))
axes[0].grid()
axes[0].set_title(r'A. P4 EEG signal')

with open('curves_8_12Hz_rand.pkl', 'rb') as handle:
    b = pickle.load(handle)

corrs_fir = b['fir']
corrs_cfir_f = b['cfir_f']
corrs_cfir_wf = b['cfir_wf']
corrs_cfir_t = b['cfir_t']
corrs_hilbert = b['hilbert']
delays = b['delays']
min_phase_delays = b['min_phase_delays']
corrs_min_phase = b['corrs_min_phase']
axes[1].plot(delays/fs*1000, corrs_fir.mean(0), label='FIR', color='k')
axes[1].fill_between(delays/fs*1000, corrs_fir.mean(0) - corrs_fir.std(0), corrs_fir.mean(0) + corrs_fir.std(0), alpha=0.3, linewidth=0., color='k')

axes[1].plot(min_phase_delays.mean(0) / fs * 1000, corrs_min_phase.mean(0), '--', label='MinPh-FIR', color='k')
axes[1].fill_between(min_phase_delays.mean(0) / fs * 1000, corrs_min_phase.mean(0) - corrs_min_phase.std(0), corrs_min_phase.mean(0) + corrs_min_phase.std(0), alpha=0.3, linewidth=0., color='k')


axes[1].plot(delays / fs * 1000, corrs_hilbert.mean(0), '-.', label='Hilbert-FFT', color='k')
axes[1].fill_between(delays / fs * 1000, corrs_hilbert.mean(0) - corrs_hilbert.std(0), corrs_hilbert.mean(0) + corrs_hilbert.std(0), alpha=0.3, linewidth=0., color='k')


axes[1].plot(delays / fs * 1000, corrs_cfir_f.mean(0), label='F-cFIR', color='r')
axes[1].fill_between(delays / fs * 1000, corrs_cfir_f.mean(0) - corrs_cfir_f.std(0), corrs_cfir_f.mean(0) + corrs_cfir_f.std(0), alpha=0.3, linewidth=0., color='r')

axes[1].plot(delays / fs * 1000, corrs_cfir_wf.mean(0), '--', label='WF-cFIR', color='r')
axes[1].fill_between(delays / fs * 1000, corrs_cfir_wf.mean(0) - corrs_cfir_wf.std(0), corrs_cfir_wf.mean(0) + corrs_cfir_wf.std(0), alpha=0.3, linewidth=0., color='r')

axes[1].plot(delays / fs * 1000, corrs_cfir_t.mean(0), label='T-cFIR', color='#01adf5')
axes[1].fill_between(delays / fs * 1000, corrs_cfir_t.mean(0) - corrs_cfir_t.std(0), corrs_cfir_t.mean(0) + corrs_cfir_t.std(0), alpha=0.3, linewidth=0., color='#01adf5')


axes[1].set_xlabel('Delay, ms')
axes[1].axvline(0, color='k', alpha=0.5)
axes[1].set_xlim(-100, 250)
axes[1].set_ylim(0.2, 1)
axes[1].grid()
axes[1].set_title('B. Gaussian white noise')

plt.legend()
plt.tight_layout()


fig.subplots_adjust(wspace=0)
plt.savefig('tradeoff.png', dpi=200)
plt.show()
