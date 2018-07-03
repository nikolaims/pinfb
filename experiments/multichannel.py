from utils.load_results import load_data
from pynfb.inlets.montage import Montage
import pylab as plt
import mne
import numpy as np
from helpers import get_XY, get_ideal_H, band_hilbert, ridge


df, fs, channels, p_names = load_data('/home/kolai/_Work/predict_eeg/results/alpha-delay-subj-13_05-16_12-25-56/experiment_data.h5')
montage = Montage(channels)

x = df.loc[df.block_name == 'FB0', channels].values[:5*60*fs]

lapl = False
if lapl:
    x = x.dot(montage.make_laplacian_proj('EEG'))
del df

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
for k in range(10-3-2):
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
    b = ridge(X_train.reshape(X_train.shape[0], -1), Y_train)

    print(time()-t)

    corrs[k * 3] = np.corrcoef(np.abs(X_train.reshape(X_train.shape[0], -1).dot(b)), np.abs(Y_train))[0,1]
    corrs[k * 3 + 1] = np.corrcoef(np.abs(X_val.reshape(X_val.shape[0], -1).dot(b)), np.abs(Y_val))[0,1]
    corrs[k * 3 + 2] = np.corrcoef(np.abs(X_test.reshape(X_test.shape[0], -1).dot(b)), np.abs(Y_test))[0,1]

    mses[k * 3] = np.mean((np.abs(X_train.reshape(X_train.shape[0], -1).dot(b)) - np.abs(Y_train))**2)
    mses[k * 3 + 1] = np.mean((np.abs(X_val.reshape(X_val.shape[0], -1).dot(b)) - np.abs(Y_val))**2)
    mses[k * 3 + 2] = np.mean((np.abs(X_test.reshape(X_test.shape[0], -1).dot(b)) - np.abs(Y_test))**2)

    print(mses)
    print(corrs)

np.save('corrs_{}.npy'.format('lapl' if lapl else 'a1a2'), corrs)
np.save('mses_{}.npy'.format('lapl' if lapl else 'a1a2'), mses)