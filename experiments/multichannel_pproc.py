import numpy as np
import pylab as plt

import pandas as pd
import seaborn as sns
sns.set_style("whitegrid")

mses_lapl = -np.load('mses_lapl.npy')

corrs_a1a2 = np.load('corrs_a1a2.npy')
corrs_lapl = np.load('corrs_lapl.npy')

mses_ones_a1a2 = -np.load('mses_one_a1a2.npy')
mses_ones_lapl = -np.load('mses_one_lapl.npy')

corrs_ones_a1a2 = np.load('corrs_one_a1a2.npy')
corrs_ones_lapl = np.load('corrs_one_lapl.npy')


df = pd.DataFrame(columns=['metric', 'montage', 'n_channels', 'value', 'data'])
for j, data in enumerate(['train', 'val', 'test']):
    df = pd.concat([df, pd.DataFrame(
        {'metric': 'mse', 'montage': 'a1a2', 'n_channels': 32, 'value': np.load('mses_a1a2.npy')[j::3], 'data': data})])
    df = pd.concat([df, pd.DataFrame(
        {'metric': 'mse', 'montage': 'lapl', 'n_channels': 32, 'value': np.load('mses_lapl.npy')[j::3], 'data': data})])
    df = pd.concat([df, pd.DataFrame(
        {'metric': 'mse', 'montage': 'a1a2', 'n_channels': 1, 'value': np.load('mses_one_a1a2.npy')[j::3], 'data': data})])
    df = pd.concat([df, pd.DataFrame(
        {'metric': 'mse', 'montage': 'lapl', 'n_channels': 1, 'value': np.load('mses_one_lapl.npy')[j::3], 'data': data})])

    df = pd.concat([df, pd.DataFrame(
        {'metric': 'corr', 'montage': 'a1a2', 'n_channels': 32, 'value': np.load('corrs_a1a2.npy')[j::3], 'data': data})])
    df = pd.concat([df, pd.DataFrame(
        {'metric': 'corr', 'montage': 'lapl', 'n_channels': 32, 'value': np.load('corrs_lapl.npy')[j::3], 'data': data})])
    df = pd.concat([df, pd.DataFrame(
        {'metric': 'corr', 'montage': 'a1a2', 'n_channels': 1, 'value': np.load('corrs_one_a1a2.npy')[j::3],
         'data': data})])
    df = pd.concat([df, pd.DataFrame(
        {'metric': 'corr', 'montage': 'lapl', 'n_channels': 1, 'value': np.load('corrs_one_lapl.npy')[j::3],
         'data': data})])

sns.factorplot('data', 'value', 'montage', df.loc[df.metric=='corr'], col='n_channels', dodge=0.2)

plt.show()