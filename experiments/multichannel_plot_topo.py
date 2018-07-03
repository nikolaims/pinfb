from utils.load_results import load_data
from pynfb.inlets.montage import Montage
import pylab as plt
import mne
import numpy as np
from helpers import get_XY, get_ideal_H, band_hilbert, ridge

from mne.viz import plot_topomap

df, fs, channels, p_names = load_data('/home/kolai/_Work/predict_eeg/results/alpha-delay-subj-13_05-16_12-25-56/experiment_data.h5')
montage = Montage(channels)

x = df.loc[df.block_name == 'FB0', channels].values[:5*60*fs]

lapl = 0
if lapl:
    x = x.dot(montage.make_laplacian_proj('EEG'))
del df

x*=1e6

x_split = np.split(x[:x.shape[0]-x.shape[0]%10], 10)

X_split = []
Y_split = []

delay = 0
n_taps = 200
band = (8, 12)
for x in x_split:
    X, Y = get_XY(x, band_hilbert(x[:, channels.index('P4')], fs, band), n_taps, delay)
    X_split.append(X)
    Y_split.append(Y)

del x


corrs = np.zeros((10-3-2)*3)
mses = np.zeros((10-3-2)*3)

corrs_one = np.zeros((10-3-2)*3)
mses_one = np.zeros((10-3-2)*3)
for k in range(5):
    X_train = np.concatenate(X_split[k:3+k])
    Y_train = np.concatenate(Y_split[k:3+k])


    X_val = X_split[k+4]
    Y_val = Y_split[k+4]

    X_test = X_split[k+5]
    Y_test = Y_split[k+5]

    print(X_train.shape, X_val.shape, X_test.shape)
    print(Y_train.shape, Y_val.shape, Y_test.shape)

    #plt.plot(np.abs(Y_test))
    #plt.show()


    from time import time
    t = time()
    b = ridge(X_train.reshape(X_train.shape[0], -1), Y_train, 1000)
    print(b.shape)
    #plt.plot(b.reshape(n_taps, -1))

    #plt.plot(np.imag(b.reshape(n_taps, X_train.shape[2])))

    #plt.show()

    #plot_topomap(np.std(b.reshape(n_taps, -1)[50:-50], 0), montage.get_pos('EEG'))
    print(time()-t)



    #plot_topomap(np.argmax(np.abs(b.reshape(n_taps, -1)[50:-50]), 0), montage.get_pos('EEG'))


    corrs[k * 3] = np.corrcoef(np.abs(X_train.reshape(X_train.shape[0], -1).dot(b)), np.abs(Y_train))[0,1]
    corrs[k * 3 + 1] = np.corrcoef(np.abs(X_val.reshape(X_val.shape[0], -1).dot(b)), np.abs(Y_val))[0,1]
    corrs[k * 3 + 2] = np.corrcoef(np.abs(X_test.reshape(X_test.shape[0], -1).dot(b)), np.abs(Y_test))[0,1]

    mses[k * 3] = np.mean((np.abs(X_train.reshape(X_train.shape[0], -1).dot(b)) - np.abs(Y_train))**2)/np.var(np.abs(Y_train))
    mses[k * 3 + 1] = np.mean((np.abs(X_val.reshape(X_val.shape[0], -1).dot(b)) - np.abs(Y_val))**2)/np.var(np.abs(Y_val))
    mses[k * 3 + 2] = np.mean((np.abs(X_test.reshape(X_test.shape[0], -1).dot(b)) - np.abs(Y_test))**2)/np.var(np.abs(Y_test))


    from time import time

    X_train = X_train[:, :, [channels.index('P4')]]
    X_test = X_test[:, :, [channels.index('P4')]]
    X_val = X_val[:, :, [channels.index('P4')]]

    t = time()
    b = ridge(X_train.reshape(X_train.shape[0], -1), Y_train, 1000)
    print(b.shape)
    print(time()-t)


    corrs_one[k * 3] = np.corrcoef(np.abs(X_train.reshape(X_train.shape[0], -1).dot(b)), np.abs(Y_train))[0,1]
    corrs_one[k * 3 + 1] = np.corrcoef(np.abs(X_val.reshape(X_val.shape[0], -1).dot(b)), np.abs(Y_val))[0,1]
    corrs_one[k * 3 + 2] = np.corrcoef(np.abs(X_test.reshape(X_test.shape[0], -1).dot(b)), np.abs(Y_test))[0,1]

    mses_one[k * 3] = np.mean((np.abs(X_train.reshape(X_train.shape[0], -1).dot(b)) - np.abs(Y_train))**2)/np.var(np.abs(Y_train))
    mses_one[k * 3 + 1] = np.mean((np.abs(X_val.reshape(X_val.shape[0], -1).dot(b)) - np.abs(Y_val))**2)/np.var(np.abs(Y_val))
    mses_one[k * 3 + 2] = np.mean((np.abs(X_test.reshape(X_test.shape[0], -1).dot(b)) - np.abs(Y_test))**2)/np.var(np.abs(Y_test))

    print(mses)
    print(corrs)
    print(mses_one)
    print(corrs_one)



np.save('corrs_{}.npy'.format('lapl' if lapl else 'a1a2'), corrs)
np.save('mses_{}.npy'.format('lapl' if lapl else 'a1a2'), mses)

np.save('corrs_one_{}.npy'.format('lapl' if lapl else 'a1a2'), corrs_one)
np.save('mses_one_{}.npy'.format('lapl' if lapl else 'a1a2'), mses_one)