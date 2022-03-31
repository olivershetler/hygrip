from .pipelines import *
import matplotlib
import matplotlib.pyplot as plt
import scipy as sp

font = {
    'family' : 'DejaVu Sans',
    'weight' : 'normal',
    'size'   : 12
}
matplotlib.rc('font', **font)

def plot_raw(ds, sbj, xlim=None, measures=None, labels=None, figsize=(6, 12)):
    """ Plots figure 2 A
    :param ds: (h5py instance) dataset
    :param sbj: (str) id of subject to plot
    :param xlim: (list) x coordinates limits
    :param measures: (list) strings of desired measures to plot
    :param labels: (list) strings of ylabels for units
    :param figsize:  (tuple) size of figure
    :return: fig and axes handles
    """
    
    if measures is None:
        measures = ["task", "frc", "emg", "eeg", "oxy", "dxy", "eog", "brt"]
        labels = {
            "task": "Task VC [pu]",
            "eeg": r"EEG [mV]",
            "oxy": "HbO [mol]",
            "dxy": "HbR [mol]",
            "emg": r"EMG [mV]",
            "frc": "Force [V]",
            "eog": r"EOG [mV]",
            "brt": "Breathing [V]",
        }

    cropper_ = StartEndCropper()
    fig, ax = plt.subplots(len(measures), 1, figsize=figsize, sharex=True)
    for i, k in enumerate(measures):
        
        if k == "task":
            x, evts, fs = cropper_(ds[sbj + "/frc"], ds[sbj + "/frc"].attrs['events'], ds.attrs["frc_sfreq"])
            x = get_task(x, evts, fs)
        else:
            x, evts, fs = cropper_(ds[sbj + "/" + k], ds[sbj + "/" + k].attrs['events'], ds.attrs[k + "_sfreq"])
            
        if k == "brt":
            filt = Filter([5, 6], ds.attrs["brt_sfreq"], gpass=3, gstop=30, filttype="butter")
            x, _, _ = filt(x, None, ds.attrs["brt_sfreq"])
            
        times = np.arange(x.shape[-1]) / fs
        # Remove offset
        if xlim is not None and k not in ["task"]:
            inds = [np.argmin(np.abs(times - t)) for t in [xlim[0], xlim[0] + 1]]
            x -= x[..., inds[0]:inds[1]].mean(-1, keepdims=True)
#         elif k == "frc":
#             # hack to present VC instead of Volts only since the period 
#             # properly done in the rest of the notebook
#             x -= np.percentile(x[..., inds[0]:inds[1]], 1)
#             x = x / ds[sbj + "/frc"].attrs["MVC"][1]
#             labels[k] = "Produced VC[pu]"
        
        # Signal time selection
        if xlim is not None:
            inds = [np.argmin(np.abs(times - t)) for t in xlim]
        else:
            inds = [0, -1]
        
        cmap = plt.get_cmap("nipy_spectral")
        norm = matplotlib.colors.Normalize(vmin=0,vmax=24)
        offset, ylims = 0, [] 
        for ch in range(x.shape[0]):
            if ch > 0:
                offset -= x[ch, inds[0]:inds[1]].min()
            else:
                ylims.append(x[ch, inds[0]:inds[1]].min())
            if k in ["eeg", "oxy", "dxy", "emg"]:
                color = cmap(norm(ch))
            else:
                color = "k"
            ax[i].plot(times, x[ch]+offset, color=color, alpha=0.75)
            offset += x[ch, inds[0]:inds[1]].max()
        ylims.append(offset)
        mn, mx = x[..., inds[0]:inds[1]].min(), x[..., inds[0]:inds[1]].max()
        for j in range(evts.shape[0]):
            ev = evts[j, 0]
            lb = int(evts[j, 1])
            ax[i].plot([ev, ev], ylims, ['r', 'b', '--k'][lb])
        ax[i].set_ylim(ylims[0]*1.1, ylims[1]*1.1)
        if labels is not None:
            ax[i].set_ylabel(labels[k])
        ax[i].ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useOffset=True)
            
    if xlim is not None:
        ax[0].set_xlim(xlim)
    ax[-1].set_xlabel("t[s]")
    
    for px in ax:
        px.yaxis.tick_right()
        px.grid(True)
    return fig, ax



def make_scalp_grid(grid, figsize=(14, 8)):
    """ Returns 2D axes handles for NIRS & EEG layout
    :param grid: (array) grid of layout
    :param figsize: (tuple) figsize
    :return: figure and axes handle
    """
    n, m = grid.shape
    fig, ax = plt.subplots(n, m, sharex=True, sharey=True, figsize=figsize)
    for i in range(n):
        for j in range(m):
            if grid[i, j] == -1:
                ax[i, j].set_visible(False)
    return fig, ax


def plot_hb(nirs, evts, fs, grid, st_err=True, figsize=(10, 6)):
    """ Plots HbO and HbR in 2D layour
    :param nirs: (array) nirs data
    :param evts: (array) nirs events
    :param fs: (float) samping frequency [Hz]
    :param grid: (array) grid layout
    :param st_err: (bool) plot standard error if `True` else standard deviation
    :param figsize: (tuple) figure size
    :return: figure and axes handle
    """
    assert len(nirs.shape) == 4, "Nirs needs to be gathered by trials."
    fig, ax = make_scalp_grid(grid, figsize=figsize)
    colors = [['#4477AA', '#117733'], ['#DDCC77', '#CC6677']]
    # Compute mean and standard deviation (or standard error) across trials
    m, s = 1e6 * nirs.mean(1), 1e6 * nirs.std(1)
    if st_err:
        s = s / np.sqrt(nirs.shape[1])
    #  Plot
    times = np.arange(nirs.shape[-1]) / fs - evts[0, 0, 0]
    for i in range(2):  # for each hand
        for j in range(2):  # for oxy/dxy
            for k in range(nirs.shape[-2] // 2):
                r, c = np.argwhere(grid == k)[0]
                if k == 0:
                    ax[r, c].plot(times, m[i, 2 * k + j], color=colors[i][j],
                                  label=['left', 'right'][i] + " hand " + ['(HbO)', '(HbR)'][j])
                else:
                    ax[r, c].plot(times, m[i, 2 * k + j], color=colors[i][j])
                ax[r, c].fill_between(times, m[i, 2 * k + j] - s[i, 2 * k + j], m[i, 2 * k + j] + s[i, 2 * k + j],
                                      color=colors[i][j], alpha=0.5)
                ax[r, c].grid(True)
                xlim, ylim = ax[r, c].get_xlim(), ax[r, c].get_ylim()
    ax[-2, 0].set_ylabel(r"$\Delta Hb~[\mu mol]$")
    ax[-1, 1].set_xlabel("t [s]")
    return fig, ax


def plot_emg_power(emg, evts, fs, face="ant", tbound=None, st_err=True, plotsd=False, figsize=(10, 6)):
    """ Computes and plots the power profile of EMG
    :param emg: (array) emg data
    :param evts: (array) emg events
    :param fs: (float) sampling frequency [Hz]
    :param face: (str) selects the EMG channels to plot. Default is "ant"erior channels
    :param tbound: (list) selects time to plot
    :param st_err: (bool) plot standard error if `True` else standard deviation
    :param plotsd: (bool) plot or not standard error or deviation
    :param figsize: (tuple) figure size
    :return: figure and axes handles
    """
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=figsize)
    times = np.arange(emg.shape[-1]) / fs - evts[0, 0, 0]
    # Compute mean and standard deviation (or standard error) accross trials
    if tbound is None:
        emg = 20 * np.log10((emg ** 2) / (emg ** 2).mean(axis=-1, keepdims=True))
    else:
        ind = [int(t * fs) for t in tbound]
        emg = 20 * np.log10((emg ** 2) / (emg[..., ind[0]:ind[1]] ** 2).mean(axis=-1, keepdims=True))
    m, s = emg.mean(1), emg.std(1)
    if st_err:
        s = s / np.sqrt(emg.shape[1])
    f = 0 if face == "ant" else 1
    for i in range(2):  # for each each anterior arm channel
        for j in range(2):  # for each hand condition
            ax[i, j].plot(times, m[j, 2 * i + f], 'gray', alpha=0.9)
            if plotsd:
                ax[i, j].fill_between(times, m[j, 2 * i + f] - s[j, 2 * i + f], m[j, 2 * i + f] + s[j, 2 * i + f],
                                      'gray', alpha=0.5)
            ax[i, j].grid(True)
            ax[i, j].set_xlabel("t [s]")
            ax[0, j].set_title(["Left", "Right"][j] + " hand trials")
        ax[i, 0].set_ylabel(["Left", "Right"][i] + " arm " + face + ". EMG [$dB_{20}$]")
    fig.tight_layout()
    return fig, ax


def plot_emg_amp(emg, evts, fs, face="ant", figsize=(10, 6)):
    """ Plots EMG amplitudes
    :param emg: (array) emg data
    :param evts: (array) emg events
    :param fs: (float) sampling frequency [Hz]
    :param face: (str) selects the EMG channels to plot. Default is "ant"erior channels
    :param figsize: (tuple) figure size
    :return: figure and axes handles
    """
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=figsize)
    times = np.arange(emg.shape[-1]) / fs - evts[0, 0, 0]
    f = 0 if face == "ant" else 1
    for i in range(2):  # for each each anterior arm channel
        for j in range(2):  # for each hand condition
            ax[i, j].plot(times, emg[j, :, 2 * i + f].T, 'k', alpha=0.8)
            ax[i, j].grid(True)
            ax[i, j].set_xlabel("t [s]")
            ax[0, j].set_title(["Left", "Right"][j] + " hand trials")
        ax[i, 0].set_ylabel(["Left", "Right"][i] + " arm " + face + ". EMG [$dB_{20}$]")
    fig.tight_layout()
    return fig, ax


def plot_brt(brt, evts, fs, figsize=(16, 4), scale=1e3):
    """ Plots breath
    :param brt: (array) breath data
    :param evts: (array) events
    :param fs: (float) sampling frequency
    :param figsize: (tuple) figure size
    :return: figure and axes handles
    """
    fig, ax = plt.subplots(1, 2, sharey=True, sharex=True, figsize=figsize)
    times = np.arange(brt.shape[-1]) / fs - evts[0, 0, 0]
    for i in range(brt.shape[0]):
        for j in range(brt.shape[1]):
            ax[i].plot(times, scale * brt[i, j, 0], 'gray', alpha=0.25)
        ax[i].set_xlabel("t [s]")
        ax[i].set_title(["left", "right"][i] + " hand")
    ax[0].set_ylabel("Chest expansion [mV]")
    return fig, ax


def plot_eog(eog, evts, fs, figsize=(16, 4)):
    """ Plots EOG
    :param eog: (array) EOG data
    :param evts: (array) events
    :param fs: (float) sampling frequency
    :param figsize: (tuple) figure size
    :return: figure and axes handles
    """
    fig, ax = plot_brt(eog, evts, fs, figsize=figsize)
    ax[0].set_ylabel("EOG [mV]")
    return fig, ax


def plot_force(force, evts, fs, perc=95, alpha=0.1, figsize=(10, 5), width_ratios=[5, 1], plot_task=False):
    """ Plots force
    :param force: (array) force data
    :param evts: (array) events
    :param fs: (float) sampling frequency [Hz]
    :param perc: (float) percentile to get representative levels of force
    :param alpha: (float) transparency of force trials
    :param figsize: (tuple) figure size in inches
    :param width_ratios: (list)  ratio of widths for time plot versus violin plots
    :param plot_task: (bool) plot task on top of produced forces
    :return: figure and axes handles
    """
    fig, ax = plt.subplots(2, 2, figsize=figsize, gridspec_kw={"width_ratios": width_ratios})
    times = np.arange(force.shape[-1]) / fs - evts[0, 0, 0]
    for i in range(force.shape[0]):
        vp = []
        #  Plot delimiters for forces
        for j, y in enumerate([0.5, 0.25]):
            ax[i, 0].plot([-6, 26], [y, y], ['--k', '--k'][j], alpha=0.9)
        #  Plot forces and gather percentiles
        for j in range(force.shape[1]):
            ax[i, 0].plot(times, force[i, j, 0].T, 'gray', alpha=alpha);
            vp.append(np.percentile(force[i, j, 0], perc))
            ax[i, 0].set_ylabel(f"{['Left', 'Right'][i]} hand VC [pu]")
        #  Horizontal delimiters for violins
        for j, y in enumerate([0.5, 0.25]):
            ax[i, 1].plot([.75, 1.25], [y, y], ['--k', '--k'][j], alpha=0.9)
        # Plot violins
        ax[i, 1].violinplot(vp)
        #  Adjust axes
        ax[i, 0].set_ylim([-0.1, 1.1])
        ax[i, 1].set_ylim([-0.1, 1.1])
        ax[i, 0].grid(True)
        ax[i, 1].grid(True)
        ax[i, 1].set_xticklabels([])
        ax[i, 1].set_yticklabels([])
    if plot_task:
        x = get_task(force, evts[0,0:1], fs)
        for i in range(2):
            ax[i,0].plot(times, x[0], "k")
    ax[1, 0].set_xlabel("t [s]")
    return fig, ax


def plot_eeg(eeg, evts, fs, grid, hand, tlim=None, flim=None, mode="specgram", figsize=(16, 8), **kwargs):
    """ Plot EEG in 2D scalp layout
    :param eeg: (array) EEG data
    :param evts: (array) events
    :param fs: (float) sampling frequency
    :param grid: (array) 2D grid layout
    :param hand: (string) "left" or "right"
    :param tlim: (list) time bounds to plot
    :param flim: (list) frequncy bounds to plot in specgram mode
    :param mode: (str)
                1. "specgram" : plots spectrogram of average signal accross trials
                2. "average"  : plots average signal accross trials
                3. "trials"   : overlay all trials
    :param figsize: (tuple) figure size
    :param kwargs: (dict) additional keyworded arguments
    :return: figure and axes handle
    """

    assert len(eeg.shape) == 4, "EEG needs to be gathered by trials."
    fig, ax = make_scalp_grid(grid, figsize=figsize)

    # Compute mean and standard deviation (or standard error) accross trials
    if mode == "average":
        eeg = 1e3 * eeg.mean(1)  # (hand, chans, times)
        units = r"dB$_{20}$"
    else:
        eeg = 1e3 * eeg  # (hand, trials, chans, times)
        units = "mV"
    lmaxis, lmins = [], []
    #  Plot
    i = 0 if hand == "left" else 1
    t_on = evts[0, 0, 0]
    times = np.arange(eeg.shape[-1]) / fs - t_on
    if mode == "specgram":
        if "window" not in kwargs.keys():
            window = ("tukey", 0.25)
        else:
            window = kwargs["window"]
        if "noverlap" not in kwargs.keys():
            noverlap = None
        else:
            noverlap = kwargs["noverlap"]
        if "scaling" not in kwargs.keys():
            scaling = None
        else:
            scaling = kwargs["scaling"]

        for ch in range(eeg.shape[2]):
            N = 0
            for j in range(eeg.shape[1]):
                f, t, Sxx = signal.spectrogram(
                    eeg[i, j, ch], fs, window=window,
                    scaling=scaling, noverlap=noverlap
                )
                t -= t_on
                if tlim is not None:
                    inds = [np.argmin(np.abs(t - tt)) for tt in tlim]
                    Sxx = Sxx[:, inds[0]:inds[1]]
                    t = t[inds[0]:inds[1]]

                if flim is not None:
                    inds = [np.argmin(np.abs(f - fx)) for fx in flim]
                    Sxx = Sxx[inds[0]:inds[1]]
                    f = f[inds[0]:inds[1]]

                if j == 0:
                    Savg, N = Sxx, N + 1
                else:
                    Savg, N = Sxx + Savg, N + 1
            Savg = Savg / N
            r, c = np.argwhere(grid == ch)[0]
            fint = sp.interpolate.interp2d(t, f, Savg, kind='cubic')
            tnew = np.arange(t[0], t[-1], .1)
            fnew = np.arange(f[0], f[-1], .1)
            Snew = fint(tnew, fnew)
            tnew, fnew = np.meshgrid(tnew, fnew)
            lmaxis.append(np.percentile(Savg, 99))
            lmins.append(np.percentile(Savg, 1))
            if "scale" in kwargs.keys() and kwargs["scale"] is not None:
                ax[r, c].pcolormesh(tnew, fnew, Snew, cmap=plt.get_cmap("coolwarm"),vmax=kwargs["scale"][1], vmin=kwargs["scale"][0])
            else:
                ax[r, c].pcolormesh(tnew, fnew, Snew, cmap=plt.get_cmap("coolwarm"))

        units = 'Frequency [Hz]'
    elif mode == "average":
        for ch in range(eeg.shape[1]):
            r, c = np.argwhere(grid == ch)[0]
            ax[r, c].plot(times, eeg[i, ch], "k")
            ax[r, c].grid(True)
    elif mode == "trials":
        for ch in range(eeg.shape[2]):
            r, c = np.argwhere(grid == ch)[0]
            ax[r, c].plot(times, eeg[i, :, ch].T, "k", alpha=0.5)
            ax[r, c].grid(True)
    else:
        raise NotImplementedError(
            f'mode {mode} not implemented. Available modes are \"specgram\", \"average\" and \"trials\".')
    
    ax[-1, 1].set_xlabel("t [s]")
    ax[-2, 0].set_ylabel(units)
    fig.suptitle(hand + " hand")
    return fig, ax
