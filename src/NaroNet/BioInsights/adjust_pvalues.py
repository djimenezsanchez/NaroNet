# -*- coding: utf-8 -*-
"""Functions for controlling the family-wise error rate (FWER).
This program code is part of the MultiPy (Multiple Hypothesis Testing in
Python) package.
Author: Tuomas Puoliväli (tuomas.puolivali@helsinki.fi)
Last modified: 27th December 2017.
License: Revised 3-clause BSD
Source: https://github.com/puolival/multipy/blob/master/fwer.py
References:
[1] Hochberg Y (1988): A sharper Bonferroni procedure for multiple tests of
    significance. Biometrika 75(4):800-802.
[2] Holm S (1979): A simple sequentially rejective multiple test procedure.
    Scandinavian Journal of Statistics 6(2):65-70.
[3] Sidak Z (1967): Confidence regions for the means of multivariate normal
    distributions. Journal of the American Statistical Association 62(318):
    626-633.
WARNING: These functions have not been entirely validated yet.
"""

import numpy as np

def bonferroni(pvals, alpha=0.05):
    """A function for controlling the FWER at some level alpha using the
    classical Bonferroni procedure.
    Input arguments:
    pvals       - P-values corresponding to a family of hypotheses.
    alpha       - The desired family-wise error rate.
    Output arguments:
    significant - An array of flags indicating which p-values are significant
                  after correcting for multiple comparisons.
    """
    m, pvals = len(pvals), np.asarray(pvals)
    return pvals < alpha/float(m)

def hochberg(pvals, alpha=0.05):
    """A function for controlling the FWER using Hochberg's procedure [1].
    Input arguments:
    pvals       - P-values corresponding to a family of hypotheses.
    alpha       - The desired family-wise error rate.
    Output arguments:
    significant - An array of flags indicating which p-values are significant
                  after correcting for multiple comparisons.
    """
    m, pvals = len(pvals), np.asarray(pvals)
    # Sort the p-values into ascending order
    ind = np.argsort(pvals)

    """Here we have k+1 (and not just k) since Python uses zero-based
    indexing."""
    test = [p <= alpha/(m+1-(k+1)) for k, p in enumerate(pvals[ind])]
    significant = np.zeros(np.shape(pvals), dtype='bool')
    significant[ind[0:np.sum(test)]] = True
    return significant

def holm_bonferroni(pvals, alpha=0.05):
    """A function for controlling the FWER using the Holm-Bonferroni
    procedure [2].
    Input arguments:
    pvals       - P-values corresponding to a family of hypotheses.
    alpha       - The desired family-wise error rate.
    Output arguments:
    significant - An array of flags indicating which p-values are significant
                  after correcting for multiple comparisons.
    """
    m, pvals = len(pvals), np.asarray(pvals)
    ind = np.argsort(pvals)
    test = [p > alpha/(m+1-k) for k, p in enumerate(pvals[ind])]

    """The minimal index k is m-np.sum(test)+1 and the hypotheses 1, ..., k-1
    are rejected. Hence m-np.sum(test) gives the correct number."""
    significant = np.zeros(np.shape(pvals), dtype='bool')
    significant[ind[0:m-np.sum(test)]] = True
    return significant

def sidak(pvals, alpha=0.05):
    """A function for controlling the FWER at some level alpha using the
    procedure by Sidak [3].
    Input arguments:
    pvals       - P-values corresponding to a family of hypotheses.
    alpha       - The desired family-wise error rate.
    Output arguments:
    significant - An array of flags indicating which p-values are significant
                  after correcting for multiple comparisons.
    """
    n, pvals = len(pvals), np.asarray(pvals)
    return pvals < 1. - (1.-alpha) ** (1./n)


from scipy.interpolate import UnivariateSpline

def _ecdf(x):
    '''no frills empirical cdf used in fdrcorrection
    '''
    nobs = len(x)
    return np.arange(1,nobs+1)/float(nobs)

multitest_methods_names = {'b': 'Bonferroni',
                           's': 'Sidak',
                           'h': 'Holm',
                           'hs': 'Holm-Sidak',
                           'sh': 'Simes-Hochberg',
                           'ho': 'Hommel',
                           'fdr_bh': 'FDR Benjamini-Hochberg',
                           'fdr_by': 'FDR Benjamini-Yekutieli',
                           'fdr_tsbh': 'FDR 2-stage Benjamini-Hochberg',
                           'fdr_tsbky': 'FDR 2-stage Benjamini-Krieger-Yekutieli',
                           'fdr_gbs': 'FDR adaptive Gavrilov-Benjamini-Sarkar'
                           }

_alias_list = [['b', 'bonf', 'bonferroni'],
               ['s', 'sidak'],
               ['h', 'holm'],
               ['hs', 'holm-sidak'],
               ['sh', 'simes-hochberg'],
               ['ho', 'hommel'],
               ['fdr_bh', 'fdr_i', 'fdr_p', 'fdri', 'fdrp'],
               ['fdr_by', 'fdr_n', 'fdr_c', 'fdrn', 'fdrcorr'],
               ['fdr_tsbh', 'fdr_2sbh'],
               ['fdr_tsbky', 'fdr_2sbky', 'fdr_twostage'],
               ['fdr_gbs']
               ]


multitest_alias = {}
for m in _alias_list:
    multitest_alias[m[0]] = m[0]
    for a in m[1:]:
        multitest_alias[a] = m[0]


def lsu(pvals, alpha=0.05,method='indep'):
    '''pvalue correction for false discovery rate

    This covers Benjamini/Hochberg for independent or positively correlated and
    Benjamini/Yekutieli for general or negatively correlated tests. Both are
    available in the function multipletests, as method=`fdr_bh`, resp. `fdr_by`.

    Parameters
    ----------
    pvals : array_like
        set of p-values of the individual tests.
    alpha : float
        error rate
    method : {'indep', 'negcorr'}
    is_sorted : bool
        If False (default), the p_values will be sorted, but the corrected
        pvalues are in the original order. If True, then it assumed that the
        pvalues are already sorted in ascending order.

    Returns
    -------
    rejected : ndarray, bool
        True if a hypothesis is rejected, False if not
    pvalue-corrected : ndarray
        pvalues adjusted for multiple hypothesis testing to limit FDR

    Notes
    -----

    If there is prior information on the fraction of true hypothesis, then alpha
    should be set to alpha * m/m_0 where m is the number of tests,
    given by the p-values, and m_0 is an estimate of the true hypothesis.
    (see Benjamini, Krieger and Yekuteli)

    The two-step method of Benjamini, Krieger and Yekutiel that estimates the number
    of false hypotheses will be available (soon).

    Method names can be abbreviated to first letter, 'i' or 'p' for fdr_bh and 'n' for
    fdr_by.
    '''
    pvals = np.asarray(pvals)
    is_sorted=False
    if not is_sorted:
        pvals_sortind = np.argsort(pvals)
        pvals_sorted = np.take(pvals, pvals_sortind)
    else:
        pvals_sorted = pvals  # alias

    if method in ['i', 'indep', 'p', 'poscorr']:
        ecdffactor = _ecdf(pvals_sorted)
    elif method in ['n', 'negcorr']:
        cm = np.sum(1./np.arange(1, len(pvals_sorted)+1))   #corrected this
        ecdffactor = _ecdf(pvals_sorted) / cm
##    elif method in ['n', 'negcorr']:
##        cm = np.sum(np.arange(len(pvals)))
##        ecdffactor = ecdf(pvals_sorted)/cm
    else:
        raise ValueError('only indep and negcorr implemented')
    reject = pvals_sorted <= ecdffactor*alpha
    if reject.any():
        rejectmax = max(np.nonzero(reject)[0])
        reject[:rejectmax] = True

    pvals_corrected_raw = pvals_sorted / ecdffactor
    pvals_corrected = np.minimum.accumulate(pvals_corrected_raw[::-1])[::-1]
    del pvals_corrected_raw
    pvals_corrected[pvals_corrected>1] = 1
    if not is_sorted:
        pvals_corrected_ = np.empty_like(pvals_corrected)
        pvals_corrected_[pvals_sortind] = pvals_corrected
        del pvals_corrected
        reject_ = np.empty_like(reject)
        reject_[pvals_sortind] = reject
        return reject_, pvals_corrected_
    else:
        return reject, pvals_corrected
