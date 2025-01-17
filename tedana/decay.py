"""
Functions to estimate S0 and T2* from multi-echo data.
"""
import logging
import scipy
import numpy as np
from tedana import utils

LGR = logging.getLogger(__name__)
RepLGR = logging.getLogger('REPORT')
RefLGR = logging.getLogger('REFERENCES')


def mono_exp(tes, s0, t2star):
    """
    Specifies a monoexponential model for use with scipy curve fitting

    Parameters
    ----------
    tes : (E,) :obj:`list`
        Echo times
    s0 : :obj:`float`
        Initial signal parameter
    t2star : :oj:`float`
        T2* parameter

    """
    return s0 * np.exp(-tes / t2star)


def fit_decay(data, tes, mask, masksum, fittype):
    """
    Fit voxel-wise monoexponential decay models to `data`

    Parameters
    ----------
    data : (S x E [x T]) array_like
        Multi-echo data array, where `S` is samples, `E` is echos, and `T` is
        time
    tes : (E,) :obj:`list`
        Echo times
    mask : (S,) array_like
        Boolean array indicating samples that are consistently (i.e., across
        time AND echoes) non-zero
    masksum : (S,) array_like
        Valued array indicating number of echos that have sufficient signal in
        given sample
    fittype : {loglin, curvefit}
        The type of model fit to use

    Returns
    -------
    t2s_limited : (S,) :obj:`numpy.ndarray`
        Limited T2* map. The limited map only keeps the T2* values for data
        where there are at least two echos with good signal.
    s0_limited : (S,) :obj:`numpy.ndarray`
        Limited S0 map.  The limited map only keeps the S0 values for data
        where there are at least two echos with good signal.
    t2ss : (S x E-1) :obj:`numpy.ndarray`
        Voxel-wise T2* estimates using ascending numbers of echoes, starting
        with 2.
    s0vs : (S x E-1) :obj:`numpy.ndarray`
        Voxel-wise S0 estimates using ascending numbers of echoes, starting
        with 2.
    t2s_full : (S,) :obj:`numpy.ndarray`
        Full T2* map. For voxels affected by dropout, with good signal from
        only one echo, the full map uses the T2* estimate from the first two
        echoes.
    s0_full : (S,) :obj:`numpy.ndarray`
        Full S0 map. For voxels affected by dropout, with good signal from
        only one echo, the full map uses the S0 estimate from the first two
        echoes.

    Notes
    -----
    1.  Fit monoexponential decay function to all values for a given voxel
        across TRs, per TE, to estimate voxel-wise :math:`S_0` and
        :math:`T_2^*`:

        .. math::
            S(TE) = S_0 * exp(-R_2^* * TE)

            T_2^* = 1 / R_2^*

    2.  Replace infinite values in :math:`T_2^*` map with 500 and NaN values
        in :math:`S_0` map with 0.
    3.  Generate limited :math:`T_2^*` and :math:`S_0` maps by doing something.
    """
    RepLGR.info("A monoexponential model was fit to the data at each voxel "
                "using log-linear regression in order to estimate T2* and S0 "
                "maps. For each voxel, the value from the adaptive mask was "
                "used to determine which echoes would be used to estimate T2* "
                "and S0.")

    if data.shape[1] != len(tes):
        raise ValueError('Second dimension of data ({0}) does not match number '
                         'of echoes provided (tes; {1})'.format(data.shape[1], len(tes)))
    elif not (data.shape[0] == mask.shape[0] == masksum.shape[0]):
        raise ValueError('First dimensions (number of samples) of data ({0}), '
                         'mask ({1}), and masksum ({2}) do not '
                         'match'.format(data.shape[0], mask.shape[0], masksum.shape[0]))

    if len(data.shape) == 3:
        n_samp, n_echos, n_vols = data.shape
        fit_data = np.mean(data, axis=2)
        fit_sigma = np.std(data, axis=2)
    else:
        n_samp, n_echos = data.shape
        n_vols = 1

    data = data[mask]
    fit_data = fit_data[mask]
    fit_sigma = fit_sigma[mask]
    t2ss = np.zeros([n_samp, n_echos - 1])
    s0vs = np.zeros([n_samp, n_echos - 1])

    for i_echo, echo_num in enumerate(range(2, n_echos + 1)):
        # perform log linear fit of echo times against MR signal
        # make DV matrix: samples x (time series * echos)
        data_2d = data[:, :echo_num, :].reshape(len(data), -1).T
        log_data = np.log(np.abs(data_2d) + 1)

        # make IV matrix: intercept/TEs x (time series * echos)
        x = np.column_stack([np.ones(echo_num), [-te for te in tes[:echo_num]]])
        X = np.repeat(x, n_vols, axis=0)
        echo_times_1d = X[:, 1] * -1

        # Log-linear fit
        betas = np.linalg.lstsq(X, log_data, rcond=None)[0]
        t2s = 1. / betas[1, :].T
        s0 = np.exp(betas[0, :]).T

        if fittype == 'curvefit':
            # perform a monoexponential fit of echo times against MR signal
            # using loglin estimates as initial starting points for fit

            fail_count = 0
            for voxel in range(t2s.size):
                try:
                    popt, cov = scipy.optimize.curve_fit(
                        mono_exp, echo_times_1d, data_2d[:, voxel],
                        p0=(s0[voxel], t2s[voxel]))
                    s0[voxel] = popt[0]
                    t2s[voxel] = popt[1]
                except RuntimeError:
                    # If curve_fit fails to converge, fall back to loglinear estimate
                    fail_count += 1
            if fail_count:
                fail_percent = 100 * fail_count / t2s.size
                LGR.debug('With {0} echoes, monoexponential fit failed on {1} ({2:.2f}%) voxel(s),'
                          ' used log linear estimate instead'.format(echo_num, fail_count,
                                                                     fail_percent))

        t2s[np.isinf(t2s)] = 500.  # why 500?
        t2s[t2s <= 0] = 1.  # let's get rid of negative values!
        s0[np.isnan(s0)] = 0.      # why 0?

        t2ss[..., i_echo] = np.squeeze(utils.unmask(t2s, mask))
        s0vs[..., i_echo] = np.squeeze(utils.unmask(s0, mask))

    # create limited T2* and S0 maps
    echo_masks = np.zeros([n_samp, n_echos - 1], dtype=bool)
    for echo in range(2, n_echos + 1):
        echo_mask = np.squeeze(echo_masks[..., echo - 2])
        echo_mask[masksum == echo] = True
        echo_masks[..., echo - 2] = echo_mask
    t2s_limited = utils.unmask(t2ss[echo_masks], masksum > 1)
    s0_limited = utils.unmask(s0vs[echo_masks], masksum > 1)

    # create full T2* maps with S0 estimation errors
    t2s_full, s0_full = t2s_limited.copy(), s0_limited.copy()
    t2s_full[masksum == 1] = t2ss[masksum == 1, 0]
    s0_full[masksum == 1] = s0vs[masksum == 1, 0]

    return t2s_limited, s0_limited, t2ss, s0vs, t2s_full, s0_full


def fit_decay_ts(data, tes, mask, masksum, fittype):
    """
    Fit voxel- and timepoint-wise monoexponential decay models to `data`

    Parameters
    ----------
    data : (S x E x T) array_like
        Multi-echo data array, where `S` is samples, `E` is echos, and `T` is
        time
    tes : (E,) :obj:`list`
        Echo times
    mask : (S,) array_like
        Boolean array indicating samples that are consistently (i.e., across
        time AND echoes) non-zero
    masksum : (S,) array_like
        Valued array indicating number of echos that have sufficient signal in
        given sample
    fittype : :obj: `str`
        The type of model fit to use

    Returns
    -------
    t2s_limited_ts : (S x T) :obj:`numpy.ndarray`
        Limited T2* map. The limited map only keeps the T2* values for data
        where there are at least two echos with good signal.
    s0_limited_ts : (S x T) :obj:`numpy.ndarray`
        Limited S0 map.  The limited map only keeps the S0 values for data
        where there are at least two echos with good signal.
    t2s_full_ts : (S x T) :obj:`numpy.ndarray`
        Full T2* timeseries.  For voxels affected by dropout, with good signal
        from only one echo, the full timeseries uses the single echo's value
        at that voxel/volume.
    s0_full_ts : (S x T) :obj:`numpy.ndarray`
        Full S0 timeseries. For voxels affected by dropout, with good signal
        from only one echo, the full timeseries uses the single echo's value
        at that voxel/volume.
    """
    n_samples, _, n_vols = data.shape
    tes = np.array(tes)

    t2s_limited_ts = np.zeros([n_samples, n_vols])
    s0_limited_ts = np.copy(t2s_limited_ts)
    t2s_full_ts = np.copy(t2s_limited_ts)
    s0_full_ts = np.copy(t2s_limited_ts)

    for vol in range(n_vols):
        t2s_limited, s0_limited, _, _, t2s_full, s0_full = fit_decay(
            data[:, :, vol][:, :, None], tes, mask, masksum, fittype)
        t2s_limited_ts[:, vol] = t2s_limited
        s0_limited_ts[:, vol] = s0_limited
        t2s_full_ts[:, vol] = t2s_full
        s0_full_ts[:, vol] = s0_full

    return t2s_limited_ts, s0_limited_ts, t2s_full_ts, s0_full_ts
