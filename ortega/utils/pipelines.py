import numpy as np
from scipy import signal
import multiprocessing
from joblib import Parallel, delayed


def get_task(force, evts, fs, tup=1.55, tdown=0.55, repeats=10):
    """ Builds task starting at times indicated in events
    :param force: (array) force data
    :param evts: (array) force events
    :param fs: (float) force sampling frequency [Hz]
    :param tup: (float) contraction time [s]
    :param tdown: (float) relaxation time [s]
    :param repeats: (int) number of repetitions per trial
    :return: (array) task
    """
    task = np.zeros((1, force.shape[-1]))
    trial = np.concatenate([.5*np.ones((1,int(tup*fs))), np.zeros((1,int(tdown*fs)))], -1)
    trial = [trial for i in range(repeats)]
    trial = np.concatenate(trial, -1)
    n = trial.shape[-1]
    times = np.arange(force.shape[-1]) / fs
    for i in range(evts.shape[0]):
        if evts[i,1] != -1:
            i0 = np.argmin(np.abs(times-evts[i,0]))
            task[:,i0:i0+n] = trial
    return task


def primes(n, threshold):
    """ Decomponses an integer in prime numbers
    :param n: (int) integer to decompose
    :param threshold: (int) maximum integer for decomposition
    :return: (list) prime factors
    """
    primfac = []
    d = 2
    while d * d <= n:
        while (n % d) == 0:
            assert d < threshold, \
                f"factors are greater than threshold {threshold}"
            primfac.append(d)
            n //= d
        d += 1
    assert n < threshold, \
        f"factors are greater than threshold {threshold}"
    if (n > 1):
        primfac.append(n)
    return primfac


def decimate(x, in_fs, out_fs, axis=-1, threshold=13):
    """ Decimates signal to out_fs avoiding downsampling factors greater than `thres`
    :param x: (array) signal to decimate
    :param in_fs: (float) input signal sampling frequency [Hz]
    :param out_fs: (float) desired output signal sampling frequency [Hz]
    :param axis: (int) axis along which decimate the signal. Last axis by default.
    :param threshold: (int) maximum decimation factor
    :return: (array) decimated signal
    """
    if in_fs != out_fs:
        assert in_fs % out_fs == 0, \
            f"input sfreq ({in_fs}Hz) should be divisible by output sfreq ({out_fs}Hz)"
        dsf = in_fs / out_fs
        while dsf % 10 == 0:
            x = signal.decimate(x, 10, axis=axis)
            dsf /= 10
        pf = primes(dsf, threshold)
        for dsf in pf:
            x = signal.decimate(x, int(dsf), axis=axis)
    return x


class Pipeline(object):
    """ Class storing the list of processes defining the pipeline and applying them when called
    """
    def __init__(self, processes, name=""):
        self._name = name
        if len(processes) == 0:
            self._processes = None
        else:
            self._processes = processes

    def __repr__(self):
        s = f"Pipeline({self._name})\n"
        if self._processes is None:
            s += f"\tplaceholder"
        else:
            for proc in self._processes:
                s += f"\t{proc.__repr__()}"
        return s

    def __call__(self, x, evts, fs, **kwargs):
        if self._processes is not None:
            for proc in self._processes:
                x, evts, fs = proc(x, evts, fs, **kwargs)
        return x, evts, fs



class EpochExtractor(object):
    """ Pipeline process
    Extract the epochs indicated by events within `tbound` time boundaries
    """
    def __init__(self, tbound):
        self._tbound = tbound

    def __repr__(self):
        return f"EpochExtractor(t_0[s]={self._tbound[0]}, t_end[s]={self._tbound[1]})\n"

    def __call__(self, x, evts, fs, **kwargs):
        events, epochs = [], []
        times = np.arange(x.shape[-1]) / fs
        inds = [int(t * fs) for t in self._tbound]
        for i in range(evts.shape[0]):
            if evts[i, 1] != -1:
                # find crop indices {t_evt+t_bound[0], t_evt+t_bound[1]}
                ind0 = int(np.argmin(np.abs(times - evts[i, 0])))
                # append cropped epoch
                epochs.append(
                    np.expand_dims(x[:, ind0 + inds[0]:ind0 + inds[1]], axis=0)
                )
                # append time shifted event
                events.append(
                    np.expand_dims(
                        np.array((-self._tbound[0], evts[i, 1])),
                        axis=0)
                )
        epochs = np.concatenate(epochs, axis=0)
        events = np.concatenate(events, axis=0)

        return epochs, events, fs



class LabelGatherer(object):
    """ Pipeline process
    Gathers epochs by "left" and "right" hand labels in first axis of array
    """
    def __init__(self, labels=[0, 1]):
        self._labels = labels

    def __repr__(self):
        return f"LabelGatherer()\n"

    def __call__(self, x, evts, fs, **kwargs):
        left_inds = evts[:, 1] == 0
        right_inds = evts[:, 1] == 1
        left = np.expand_dims(x[left_inds], axis=0)
        right = np.expand_dims(x[right_inds], axis=0)
        left_evts = np.expand_dims(evts[left_inds], axis=0)
        right_evts = np.expand_dims(evts[right_inds], axis=0)

        x = np.concatenate([left, right], axis=0)
        evts = np.concatenate([left_evts, right_evts], axis=0)

        return x, evts, fs



class Filter(object):
    """ Pipeline process
    Instantiates filter and filters signal when called
    """
    def __init__(self, bands, fs, gpass=3, gstop=60, filttype="butter"):

        self._fs = fs
        self._filttype = filttype
        self._gpass, self._gstop = gpass, gstop

        if self._filttype != "notch":
            if type(bands[0]) == list:
                assert bands[0][0] > bands[1][0] and bands[0][1] < bands[1][1], \
                    f"Bandpass {bands[0]} Hz should be contained in bandstop {bands[1]} Hz."
                self._wp = [f / (fs / 2) for f in bands[0]]
                self._ws = [f / (fs / 2) for f in bands[1]]
                self._btype = "bandpass"
            else:
                self._wp = bands[0] / (fs / 2)
                self._ws = bands[1] / (fs / 2)
                self._btype = "lowpass" if self._wp < self._ws else "highpass"
        else:
            self._f0 = bands[0]  # central freq
            self._Q = bands[1]  #  Q = ( f0 / (fs / 2) ) / bw -> -3dB bandwidth
            self._btype = "bandstop"

        if self._filttype == "butter":
            self._n, self._wn = signal.buttord(self._wp, self._ws, gpass, gstop)
            self._ba = signal.butter(self._n, self._wn, btype=self._btype)
        elif self._filttype == "ellip":
            self._n, self._wn = signal.ellipord(self._wp, self._ws, gpass, gstop)
            self._ba = signal.ellip(self._n, 1, 10, self._wn, btype=self._btype)
        elif self._filttype == "notch":
            self._ba = signal.iirnotch(self._f0, self._Q, self._fs)
        else:
            raise NotImplementedError(f"{self._filttype} not implemented")

        if not np.all(np.abs(np.roots(self._ba[1])) < 1):
            raise ArithmeticError(f"unstable filter, denominator roots bigger than 1")

    def __repr__(self):
        if self._filttype != "notch":
            if type(self._wn) == np.ndarray or type(self._wn) == list:
                fn = [w * (self._fs / 2) for w in self._wn]
                fn = f"[{fn[0]: 2.2f}, {fn[1]: 2.2f}]"
            else:
                fn = self._wn * (self._fs / 2)
                fn = f"{fn: 2.2f}"
            s = f"Filter(type={self._filttype}-{self._btype}, order={self._n}, sfreq[Hz]={self._fs}, fn[Hz]={fn}, gpass[dB]={self._gpass}, gstop[dB]={self._gstop})\n"
        else:
            s = f"Filter(type={self._filttype}-{self._btype}, sfreq[Hz]={self._fs}, fn0[Hz]={self._f0: 2.2f}, Q[1]={self._Q})\n"
        return s

    def __call__(self, x, evts, fs, **kwargs):
        assert fs == self._fs, \
            f"Signal sfreq {fs} Hz different from process sfreq {self._fs} Hz"
        x = signal.filtfilt(self._ba[0], self._ba[1], x)
        if (np.isnan(x).any()):
            raise ValueError("filter returned NaN values")
        if (np.isinf(x).any()):
            raise ValueError("filter returned Inf values")
        return x, evts, fs



class Downsampler(object):
    """ Pipeline process
    Downsample to specified `out_fs` sampling frequency
    """
    def __init__(self, in_fs, out_fs, axis=-1, threshold=13):
        assert in_fs % out_fs == 0, \
            f"input sfreq ({in_fs}Hz) should be divisible by output sfreq ({out_fs}Hz)"
        self._in_fs = in_fs
        self._out_fs = out_fs
        self._axis = axis
        self._threshold = threshold

    def __repr__(self):
        return f"Downsampler(sfreq_in[Hz]={self._in_fs}, sfreq_out[Hz]={self._out_fs}, axis={self._axis}, thres={self._threshold})\n"

    def __call__(self, x, evts, fs, **kwargs):
        assert self._in_fs == fs, \
            f"x sfreq ({fs}Hz) different from expected sfreq ({self._in_fs}Hz)"
        x = decimate(x, fs, self._out_fs, axis=self._axis, threshold=self._threshold)
        return x, evts, self._out_fs


class VCconverter(object):
    """ Pipeline process
    Converts force in V to Voluntary Contraction per unit (VC[pu])
    Notes
        - Force in V needs to be filtered to remove offset
        - Requires kwarg `mvc` with Maximum Voluntary Contraction in V available in dataset
    """
    def __repr__(self):
        return f"VCconverter()\n"

    def __call__(self, x, evts, fs, **kwargs):
        assert len(x.shape) == 4, "Input needs to be gathered by label"
        for i in range(x.shape[0]):
            x[i] = x[i] / kwargs["mvc"][i]
        return x, evts, fs



class HbConverter(object):
    """ Pipeline process
    Converts light intensity in Oxy and Deoxy Hemoglobin concentration ([HbO] and [HbR])
    Notes:
        - Parameters may be introduced as a list in the same order and units as specified in `ds.attrs["nirs_..."]`
          or as text in the same format
        - Compared to other processes, this one takes 2 signal inputs and returns single array with concatenated
          HbO and HbR measures
    """
    def __init__(self, DPF, SD, exc, tbound=None):
        self._DPF = self.__parse__(DPF)
        self._SD = self.__parse__(SD)
        self._exc = self.__parse__(exc)
        self._tbound = tbound

    @staticmethod
    def __parse__(param):
        if type(param) == np.bytes_:
            param = param.decode("utf-8")
        if type(param) == str:
            param = param.split(",")
            param = [p.split("=")[-1] for p in param]
            param = [float(p) for p in param]
        elif type(param) != list:
            raise NotImplementedError(f"param has to be of type np.bytes_, str or list but is {type(param)}")

        if len(param) == 2:  # DPF
            param = np.array(param).transpose()
        elif len(param) == 4:  # exc
            param = np.reshape(np.array(param), (2, 2))
        elif len(param) == 1:  # SD
            param = param[0]
        return param

    def __repr__(self):
        return f"HbConverter(tbound[s]={self._tbound}, DPF[1]={self._DPF}, SD[cm]={self._SD}, exc[cm^-1 mol^-1]={self._exc.reshape(-1)})\n"

    def __call__(self, x, evts, fs, **kwargs):

        assert (x > 0).all(), "Optical intensity must be strictly positive."
        # Converts to Optical Density (OD)
        if self._tbound is not None:
            ind = [int(t * fs) for t in self._tbound]
        else:
            ind = [0, -1]
        x = -np.log10(x / np.mean(x[..., ind[0]:ind[1]], axis=-1, keepdims=True))
        wl1, wl2 = np.split(x, 2, axis=-2)
        x = []
        # Apply Beer-Lambert law
        C = np.linalg.pinv(self._SD * self._DPF * self._exc)
        for i in range(wl1.shape[-2]):
            x.append(
                np.matmul(
                    C, np.concatenate(
                        (wl1[..., i:i + 1, :], wl2[..., i:i + 1, :]),
                        axis=-2
                    )
                )
            )
        x = np.concatenate(x, axis=-2)
        return x, evts, fs


class Joiner(object):
    """ Pipeline process
    Joins `x` input signal with additonal keyworded argument `y` signal along declared axis
    """
    def __init__(self, axis):
        self._axis = axis

    def __repr__(self):
        return f"Joiner(axis={self._axis})\n"

    def __call__(self, x, evts, fs, **kwargs):
        assert "y" in kwargs.keys(), "Joiner requires additional signal as kwarg \"y\""
        x = np.concatenate((x, kwargs["y"]), axis=self._axis)
        return x, evts, fs



class BaselineRemover(object):
    """ Pipeline process
    Removes baseline by substracting signal average between specified time bound `tbound`
    Bound is specified with time starting at 0 at the beginning of the signal to correct
    """

    def __init__(self, tbound):
        self._tbound = tbound

    def __repr__(self):
        return f"BaselineRemover(tbound[s]={self._tbound})\n"

    def __call__(self, x, evts, fs, **kwargs):
        inds = [int(t * fs) for t in self._tbound]
        baseline = x[..., inds[0]:inds[1]].mean(-1, keepdims=True)
        x = x - baseline
        return x, evts, fs


class MeanReferencer(object):
    """ Pipeline process
    Removes accross channels mean of a set of measures
    """
    def __init__(self, axis=-2):
        self._axis = axis

    def __repr__(self):
        return f"MeanReferencer(axis={self._axis})\n"

    def __call__(self, x, evts, fs, **kwargs):
        x /= x.std(-1, keepdims=True)
        x -= x.mean(self._axis, keepdims=True)
        x -= x.mean(-1, keepdims=True)
        x /= x.std(-1, keepdims=True)
        return x, evts, fs


class StartEndCropper(object):
    """ Pipeline process
    Crops a signal from the synchronised beginning to the synchronised end of the recording
    """
    def __repr__(self):
        return f"StartEndCropper()\n"

    def __call__(self, x, evts, fs, **kwargs):
        times = np.arange(x.shape[-1]) / fs
        # find inds to first and last trigger
        inds = [np.argmin(np.abs(times - evts[i, 0])) for i in [0, -1]]
        # crop to inds
        t_on = evts[0,0]
        x, evts = x[:, inds[0]:inds[1]], evts[1:-1, :]
        evts[:, 0] -= t_on
        return x, evts, fs



class EegPipe(object):
    """ Superclass making parallel processing of a pipeline across subjects
    To use as inspiration for subjects parallel processing
    """
    def __init__(self, dataset, eeg_fs=250):
        self._num_cores = multiprocessing.cpu_count()
        pipe = []
        pipe.append(Downsampler(dataset.attrs["eeg_sfreq"], eeg_fs))
        pipe.append(EpochExtractor([-5, 25]))
        pipe.append(LabelGatherer())
        self._pipeline = Pipeline(pipe, "eeg")

    def __repr__(self):
        s = "EegPipe:\n"
        for p in self._pipeline._processes:
            s += "\t" + p.__repr__()
        return s

    def __process__(self, eeg, evts, fs):
        eeg, evts, fs = self._pipeline(eeg, evts, fs)
        return eeg, evts, fs

    def __call__(self, dataset, subjects, *args, **kwargs):

        output = Parallel(n_jobs=self._num_cores)(
            delayed(self.__process__)
                (
                dataset[subject + "/eeg"][()],
                dataset[subject + "/eeg"].attrs["events"],
                dataset.attrs["eeg_sfreq"],
            ) for subject in subjects
        )
        axis = 0 if len(output[0][0].shape) == 2 else 1
        eeg, evts, fs = [], [], []
        print(output[0][1].shape)
        for sbj in output:
            if axis == 0:
                eeg.append(np.expand_dims(sbj[0], 0))
                evts.append(np.expand_dims(sbj[1], 0))
            else:
                eeg.append(sbj[0])
                evts.append(sbj[1])
        eeg = np.concatenate(eeg, axis=axis)
        evts = np.concatenate(evts, axis=axis)
        fs = output[0][2]
        return eeg, evts, fs