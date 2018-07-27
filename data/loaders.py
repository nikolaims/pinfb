import numpy as np
import pandas as pd


def generate_data(fs, n_samples, x_type='rand'):
    t = np.arange(n_samples) / fs
    if x_type == 'saw':
        x = np.sin(2 * np.pi * 10 * t) * (np.arange(n_samples) % (fs * 2))
    elif x_type == 'rand':
        x = np.random.normal(size=n_samples)
    elif x_type == 'eeg':
        df = pd.read_csv(r'/home/nikolai/_Work/alpha_delayed_nfb/p4-nfb-pilot_02-06_14-28-12/p4a2-nfb-pilot.csv')
        start = 0
        x = df['P4'].as_matrix()[start:start + n_samples]
        x = (x - np.mean(x)) / np.std(x)
    else:
        raise TypeError('Unknown data type')
    return x

def load_alpha_delay_subjects(subj=1):

    df = pd.read_pickle('/home/kolai/Data/pred_eeg/s{}-fb.pkl'.format(subj))
    fb_types = ['FB0', 'FB500', 'FB1000', 'FBMock']
    x =  df.loc[df.block_name.isin(fb_types), 'P4'].as_matrix()
    mask = np.where(x>np.std(x)*7)[0]

    for k in mask:
        x[k:k+2000] = np.nan

    x = x[~np.isnan(x)]
    return x

if __name__ == '__main__':
    import pylab as plt
    #plt.plot(generate_data(500, 1000, 'eeg'))

    plt.plot(load_alpha_delay_subjects(1))
    plt.show()