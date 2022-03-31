from sklearn.decomposition import FastICA
import warnings
from .pipelines import *

def compute_ica(x, n_components=None, tol=1e-3, max_iter=500):
    """ Computes sources and mixing matrices
        ** thanks to ** scikit-learn.org
    :param x: (array) shape (num_samps, num_chans)
    :param n_components: (int) number of component to use in decomposition
    :param tol: (float) tolerance in convergence
    :param max_iter: (int) max number of iterations
    :return: (tuple) sources, mixing, unmixing mean and bool for convergence


    """
    assert len(x.shape) == 2, f"x is {len(x.shape)}D but only 2D is accepted"
    assert x.shape[0] > x.shape[1], f"x shape is {x.shape} but time axis should be last"
    #  num_components, num_channels by default
    if n_components is None:
        n_components = x.shape[1]
    ica = FastICA(n_components=n_components, whiten=True, max_iter=max_iter, tol=tol)

    converged = True
    with warnings.catch_warnings(record=True)as w:
        S_ = ica.fit_transform(x)  # (num_samps, num_comps)
        if len(w) == 1 and "did not converge" in str(w[-1].message):
            converged = False
    # mixing_matrix = pinv(components): sources (-> white data) -> sources
    M_ = ica.mixing_  # (num_chans, num_comps)
    # pinv(np.dot(unmixing_matrix, self.whitening_)) : data (-> white_data) -> sources
    U_ = ica.components_  # (num_comps, num_chans)
    #  mean over features to undo whitening
    mean_ = ica.mean_  # (num_chans,)
    return S_, M_, U_, mean_, converged


def align(x, y, fs, threshold=1):
    """ Aligns to signals with same sfreq in time
    :param x: (array) shape (tsamps,)
    :param y: (array) shape (tsamps',)
    :param fs: (float) sampling frequency [Hz]
    :param threshold: (float) maxmimum delay possible in [s]
    :return:
    """
    assert len(x.shape) == 1 and len(y.shape) == 1, \
        f"align only accepts 1D vectors but got {len(x.shape)} and {len(y.shape)}"
    min_len = np.min([x.shape[0], y.shape[0]])
    x, y = x[:min_len], y[:min_len]
    cor = signal.correlate(x, y, mode="full")
    # rectify (signals can be of opposite sign only)
    cor = np.abs(cor)
    mx_ind = np.argmax(cor)
    delay = mx_ind - cor.shape[0] // 2
    if np.abs(delay * fs) <= threshold:
        if delay > 0:
            x, y = x[delay:], y[:-delay]
        elif delay < 0:
            x, y = x[:delay], y[-delay:]
    return x, y


def reject(S_, ref, max_num, threshold, fs, _debug=False):
    """ Find estimated sources to reject based on `ref` signals. Signals should be high pass filtered
        Rejects the max_num of sources that are most correlated to any of the ref signals
    :param S_: (array) sources of shape (num_samps, num_comps)
    :param ref: (array) confound reference of shape (num_samps')
    :param max_num: (int) maximum number of source to remove
    :param threshold: (float) correlation with reference threshold above which accept rejection
    :param fs: (float) sampling frequency [Hz]
    :param _debug: (bool) If `True` plots sources and reference
    :return: (tuple) list of rejected sources and correlations of sources with references
    """
    assert len(S_.shape) == 2, f"`S_` should be 2D array but found {len(S_.shape)} dims"
    assert len(ref.shape) == 2, f"`ref` should be 2D array but found {len(ref.shape)} dims"

    pears = []
    for i in range(S_.shape[1]):
        paux = []
        for j in range(ref.shape[0]):
            # avoid operations overwriting memory
            x_, y_ = np.copy(S_[:, i]), np.copy(ref[j])
            # standardise signals (should have been highpass filtered already)
            x_, y_ = x_ / x_.std(), y_ / y_.std()
            # align
            x_, y_ = align(x_, y_, fs, threshold)
            # compute pearson correlation
            paux.append(np.corrcoef(x_, y_)[0, 1])
        # select max correlation to any reference
        p = max(paux)
        pears.append(p)

    pears = np.abs(np.array(pears))

    # select max_num of sources with highest correlation above threshold
    rejected = []
    max_inds = list(pears.argsort()[-max_num:][::-1])
    for i in max_inds:
        if pears[i] > threshold:
            rejected.append(i)

    if _debug and len(rejected) > 0:
        plt.plot(S_[:, rejected[0]], alpha=0.5)
        plt.plot(ref.T, alpha=0.2)
        plt.gcf().set_size_inches(16, 4)
        plt.show()

    return rejected, pears


def reconstruct(x, M_, U_, mean_, reject):
    """ Reconstruct original signal without rejected estimated sources
        Data doesn't need to be standardised since U_/M_ already whiten/color the signals
    :param x: (array) original signal of shape (num_samps, num_chans)
    :param M_: (array) mixing matrix of shape (num_chans, num_comps)
    :param U_: (array) unmixing matrix of shape (num_comps, num_chans) == pinv(np.dot(unmixing_matrix, self.whitening_))
    :param mean_: (array) means for coloring, shape (num_chans,)
    :param reject: (list) components to zero in the recovered sources
    :return: (array) reconstructed signals
    """
    assert type(reject) == list, \
        f"rejected_components is type {type(reject)} but can only be list"
    # Estimate sources
    S_ = np.dot(x, U_.T)  # (num_samps, num_comps)

    # zero rejected recovered sources
    for i in reject:
        S_[:, i] = 0
    rec = np.dot(S_, M_.T) + mean_  # (num_samps, num_chans)
    return rec


class IcaReconstructor(object):
    """ Pipeline process
    ICA gives better results when there are more samples. The `gathered` mode concatenates samples per channel.
    On the other if there are many sources along a recording this might be less well captured. If we guess there are
    different sources for different conditions or at different recording times selecting the time-periods (`tlim`) to
    apply the ICA to might improve results.
    """

    def __init__(self, fs, n_components=None, max_components=2, threshold=0.3, tolerance=1e-2, max_iter=500,
                 prefilter=None, tlim=None, rec_tlim=None, mode="continuous", standarize=False, verbose=True):

        self._fs = fs
        self._n_components = n_components
        self._max_components = max_components
        self._threshold = threshold
        self._prefilter = prefilter
        self._cropper = StartEndCropper()
        self._tol = tolerance
        self._max_iter = max_iter
        if mode == "per_epoch":
            self._tlim = tlim
            self._rec_tlim = rec_tlim
        else:
            self._tlim = None
            self._rec_tlim = None
        self._stdze = standarize
        self._mode = mode
        self._verbose = verbose
        if n_components is not None:
            assert max_components <= n_components, \
                "max_num should be greater than n_components"

    def __repr__(self):
        if self._n_components is None:
            n = '\"as channels\"'
        else:
            n = str(self._n_components)
        return "IcaPre" + self._prefilter.__repr__() + \
               f"\tIcaReconstructor(sfreq[Hz]={self._fs}, n_comps={n}, max={self._max_components}," + \
               f" thres={self._threshold}, tol={self._tol}, max_iter={self._max_iter}, tlim={self._tlim}," + \
               f" mode={self._mode}, standarize={self._stdze})\n"

    def __call__(self, x, evts, fs, **kwargs):
        assert "ref" in kwargs, \
            'IcaReconstructor requires a reference signal with key \"ref\"'
        assert "ref_fs" in kwargs, \
            'IcaReconstructor requires a reference sampling frequency with key \"ref_fs\"'
        assert "ref_evts" in kwargs, \
            'IcaReconstructor requires reference events with key \"ref_evts\"'
        ref, ref_fs, ref_evts = kwargs["ref"], kwargs["ref_fs"], kwargs["ref_evts"]

        assert len(x.shape) == len(ref.shape), \
            f"reference signal ({len(ref.shape)}D) and input ({len(x.shape)}D) should have undergone same epoching"
        if self._mode == "per_epoch":
            assert len(x.shape) > 2, \
                f"invalid input shape ({len(x.shape)}D) for \'per_epoch\' mode"
        if self._mode == "per_condition":
            assert len(x.shape) == 4, \
                f"invalid input shape ({len(x.shape)}D) for \'per_condition\' mode"

        #  copy to not corrupt originals
        x_, ref_ = np.copy(x), np.copy(ref)
        evts_, ref_evts_ = np.copy(evts), np.copy(ref_evts)
        fs_, ref_fs_ = np.copy(fs), np.copy(ref_fs)

        # crop start to end of recordings
        if len(x.shape) == 2:
            x_, evts_, fs_ = self._cropper(x_, evts_, fs_)
            ref_, ref_evts_, ref_fs_ = self._cropper(ref_, ref_evts_, ref_fs_)

        # check input is numpy array so we can write on it
        if type(x) != np.array:
            x = x[()]

        # prestandarise in time axis
        if self._stdze:
            x_ /= x_.std(-1, keepdims=True)
            ref_ /= ref_.std(-1, keepdims=True)
            x /= x.std(-1, keepdims=True)

        #  decimate to ICA sfreq
        x_ = decimate(x_, fs, self._fs, axis=-1)
        ref_ = decimate(ref_, ref_fs, self._fs, axis=-1)

        #  apply filter to center and remove offset
        if self._prefilter is not None:
            x_, _, fs_ = self._prefilter(x_, None, self._fs)
            ref_, _, ref_fs_ = self._prefilter(ref_, None, self._fs)

        #  Indices for ICA computation
        t_on = evts[0, 0, 0]
        times_ = np.arange(x_.shape[-1]) / self._fs - t_on
        if self._tlim is None:
            inds = [0, -1]
        else:
            inds = [np.argmin(np.abs(times_ - t)) for t in self._tlim]

        # Indices for ICA reconstruction
        if self._rec_tlim is None:
            inds_rec = [0, -1]
        else:
            inds_rec = [np.argmin(np.abs(times_ - t)) for t in self._rec_tlim]

        reshape_x = False
        if self._mode == "per_condition":
            # crop to desired time section
            x_, ref_ = x_[..., inds[0]:inds[1]], ref_[..., inds[0]:inds[1]]
            # serialize (conds, chans, times * trials)
            x_ = np.moveaxis(x_, 1, -1)
            ref_ = np.moveaxis(ref_, 1, -1)
            x_ = x_.reshape((x_.shape[0], x_.shape[1], -1), order='F')
            ref_ = ref_.reshape((ref_.shape[0], ref_.shape[1], -1), order='F')

            for i in range(x_.shape[0]):
                #  compute ICA
                S_, M_, U_, mean_, conv = compute_ica(x_[i].T, n_components=self._n_components, tol=self._tol,
                                                      max_iter=self._max_iter)
                rejected, pears = reject(S_, ref_[i], self._max_components, self._threshold, self._fs)

                # Reconstruct original epochs on desired time
                for j in range(x.shape[1]):
                    x[i, j, :, inds_rec[0]:inds_rec[1]] = reconstruct(x[i, j, :, inds_rec[0]:inds_rec[1]].T, M_, U_,
                                                                      mean_, rejected).T
                if self._verbose:
                    print(f"\t cond {i}: conv {conv}: {len(rejected)} rejected component(s) above corrcoef_thres = {self._threshold} : {pears[rejected]}")

        elif self._mode in ["continuous", "per_epoch"]:
            if len(x_.shape) == 4:
                x_ = np.concatenate([x_[0], x_[1]], 0)
                ref_ = np.concatenate([ref_[0], ref_[1]], 0)

            if len(x.shape) == 4:
                x = np.concatenate([x[0], x[1]], 0)
                reshape_x = True

            if self._mode == "continuous":
                if len(x_.shape) == 3:
                    # crop to desired time section
                    t_on = evts[0, 0, 0] if len(evts_.shape) == 3 else evts[0, 0]
                    times_ = np.arange(x_.shape[-1]) / self._fs - t_on
                    if self._tlim is None:
                        inds = [0, -1]
                    else:
                        inds = [np.argmin(np.abs(times_ - t)) for t in self._tlim]
                    x_, ref_ = x_[..., inds[0]:inds[1]], ref_[..., inds[0]:inds[1]]
                    # serialize (chans, times * trials * conds)
                    x_ = np.moveaxis(x_, 0, -1)
                    ref_ = np.moveaxis(ref_, 0, -1)
                    x_ = x_.reshape((x_.shape[0], -1), order='F')
                    ref_ = ref_.reshape((ref_.shape[0], -1), order='F')

                #  compute ICA
                S_, M_, U_, mean_, conv = compute_ica(
                    x_.T, n_components=self._n_components, tol=self._tol, max_iter=self._max_iter
                )
                rejected, pears = reject(S_, ref_, self._max_components, self._threshold, self._fs)

                # Reconstruct original...
                if len(x.shape) == 2:  #  ... continuous recording
                    x = reconstruct(x.T, M_, U_, mean_, rejected).T
                else:  # ... epochs
                    for i in range(x.shape[0]):
                        x[i, ..., inds_rec[0]:inds_rec[-1]] = reconstruct(x[i, ..., inds_rec[0]:inds_rec[-1]].T, M_, U_,
                                                                          mean_, rejected).T
                if self._verbose:
                    print(f"\tconv {conv} : {len(rejected)} rejected component(s) above corrcoef_thres = {self._threshold} : {pears[rejected]}")

            elif self._mode == "per_epoch":
                for i in range(x.shape[0]):
                    S_, M_, U_, mean_, conv = compute_ica(x_[i].T, n_components=self._n_components, tol=self._tol,
                                                          max_iter=self._max_iter)
                    rejected, pears = reject(S_, ref_[i], self._max_components, self._threshold, self._fs)
                    if self._verbose:
                        print(f"\tepoch {i} : conv {conv} : {len(rejected)} rejected component(s) above corrcoef_thres = {self._threshold} : {pears[rejected]}")
                    x[i, ..., inds_rec[0]:inds_rec[-1]] = reconstruct(
                        x[i, ..., inds_rec[0]:inds_rec[-1]].T, M_, U_, mean_, rejected
                    ).T
        else:
            raise NotImplementedError(f"{self._mode} not implemented")

        if reshape_x:
            x0, x1 = np.split(x, 2, axis=0)
            x0, x1 = np.expand_dims(x0, 0), np.expand_dims(x1, 0)
            x = np.concatenate((x0, x1), 0)

        return x, evts, fs