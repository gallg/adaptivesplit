import numpy as np
import operator
import warnings
from numbers import Number
from .utils import statfun_as_callable, _optional_import_
from .learning_curve import LearningCurveFit
from .resampling import PermTest, Resample

"""
# bootstrap algorithm:
 - sample with replacement: n subject, n target value
 - calculate p-value on the sample
 - power: fraction of p-values lower than alpha
 
 Advantage: less assumptions
 Disadvantage: Bootstrap is only possible if X and y is put in.
 
# Monte-carlo method:
 - assume a null distribution (e.g. normal)
 - estimate effect size based on score value (different for each score)
 - do MC simulations
 - calculate po-values
 - power: fraction of p-values lower than alpha
 
 Advantage: works when only learning curve data is present
 Disadvantage: strong assumptions
"""

scipy_pearsonr = _optional_import_("scipy.stats", "pearsonr", "scipy")
scipy_spearmanr = _optional_import_("scipy.stats", "spearmanr", "scipy")
scipy_kendalltau = _optional_import_("scipy.stats", "kendalltau", "scipy")

# Statistic functions for power calculation
# dictionary to hold all functions
_stat_funs = {}


def as_stat_fun(f):
    _stat_funs[f.__name__.split('_statfun_')[1]] = f


@as_stat_fun
def _statfun_permtest(permtest_statfun, alternative="greater", num_perm=500, n_jobs=1, **kwargs):
    score_fun = statfun_as_callable(permtest_statfun)

    if alternative == "greater" or alternative == "two_sided":
        compare = operator.ge
    elif alternative == "less":
        compare = operator.le
    else:
        raise AttributeError("alternative can be any of ['greater', 'less', 'two_sided']")

    if alternative == 'two_sided':
        def stat_fun(x, y, **kwargs):
            return np.abs(score_fun(x, y, **kwargs))
    else:
        def stat_fun(x, y, **kwargs):
            return score_fun(x, y, **kwargs)

    def power_stat_fun(x, y, **kwargs):
        kwargs.setdefault('verbose', False)
        return PermTest(stat_fun, num_samples=num_perm,
                        n_jobs=n_jobs, compare=compare).test(x, y, **kwargs).p_value

    power_stat_fun.__name__ = 'permtest_' + permtest_statfun.__name__ + '_' + alternative

    return power_stat_fun


@as_stat_fun
def _statfun_scipystats_custom(scipy_stats_fun, alternative='greater', **kwargs):
    # simply return the second value, which is the p-value
    def stat_fun(x, y):
        if len(y) > 0 and (np.all(x[0] == x[:]) or np.all(y[0] == y[:])):
            # this is to trick ConstantInputWarning #todo: more elegant solution?
            return 1
        stat, p_val = scipy_stats_fun(x, y, **kwargs)  # return the p-value only
        if alternative == 'greater':
            if stat > 0:
                p_val = p_val / 2
            else:
                p_val = 1 - p_val / 2
        elif alternative == 'less':
            if stat < 0:
                p_val = p_val / 2
            else:
                p_val = 1 - p_val / 2
        elif alternative != 'two_sided':
            raise AttributeError("alternative can be any of ['greater', 'less', 'two_sided']")

        return p_val

    stat_fun.__name__ = scipy_stats_fun.__name__ + '_' + alternative
    return stat_fun


@as_stat_fun
def _statfun_pearsonr(alternative='greater', **kwargs):
    print(scipy_pearsonr)
    return _stat_funs['scipystats_custom'](scipy_pearsonr, alternative=alternative, **kwargs)


@as_stat_fun
def _statfun_spearmanr(alternative='greater', **kwargs):
    return _stat_funs['scipystats_custom'](scipy_spearmanr, alternative=alternative, **kwargs)


@as_stat_fun
def _statfun_kendalltau(alternative='greater', **kwargs):
    return _stat_funs['scipystats_custom'](scipy_kendalltau, alternative=alternative, **kwargs)


# add extra statfuns with the @as_stat_fun annotation


def get_power_stat_funs():
    return list(_stat_funs.keys())


def create_power_stat_fun(type, *args, **kwargs):
    return _stat_funs[type](*args, **kwargs)


class _PowerEstimatorBase:

    def __init__(self, power_stat_fun,
                 stratify=None,
                 total_sample_size=None,
                 alpha=0.05,
                 n_jobs=None,
                 verbose=True,
                 message='Estimating Power'):

        self.power_stat_fun = power_stat_fun
        self.stratify = stratify
        self.sample_size = None
        self.total_sample_size = total_sample_size
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.message = message

    def _get_power_sample_size(self, sample_size):
        if self.total_sample_size is not None:
            return self.total_sample_size - np.array(sample_size)
        else:
            return sample_size

    def _update_defaults(self, **kwargs):
        for attr, value in kwargs.items():
            if value:
                setattr(self, attr, value)

    def estimate(self, y_true, y_pred, sample_size, stratify=None, power_statfun=None,
                 random_sate=None, n_jobs=None, verbose=None, message=None):
        self._update_defaults(n_jobs=n_jobs, verbose=verbose, message=message,
                              power_statfun=power_statfun, stratify=stratify)
        self.sample_size = self._get_power_sample_size(sample_size)
        pass

    def __str__(self):
        return "power of " + self.power_stat_fun.__name__


class PowerEstimatorBootstrap(_PowerEstimatorBase):
    """Calculate power using bootstrap.

    Args:
        power_stat_fun (callable): 
            Statistical function used to calculate power.
        stratify (int): 
            For classification tasks. If not None, use stratified sampling to account for class labels imbalance. 
            Defaults to None.
        total_sample_size (int): 
            The total number of samples in the data given as input. Defaults to None.
        alpha (float): 
            Statistical threshold used to calculate power. Defaults to 0.001.
        bootstrap_samples (int): 
            Number of samples selected during bootstrapping. Defaults to 100.
        n_jobs (int): 
            Number of jobs to run in parallel. Defaults to None. Power calculations are parallelized 
            over the cross-validation splits. None means 1 unless in a joblib.parallel_backend context.
            -1 means using all processors.
        verbose (bool): 
            Prints progress. Defaults to True.
        message (str, optional): 
            Message shown when power estimation starts. Defaults to 'Estimating Power with bootstrap'.
    """
    
    def __init__(self, power_stat_fun,
                 stratify=None,
                 total_sample_size=None,
                 alpha=0.001,
                 bootstrap_samples=100,
                 n_jobs=None,
                 verbose=True,
                 message='Estimating Power with bootstrap'):
        super().__init__(power_stat_fun=power_stat_fun,
                         stratify=stratify,
                         total_sample_size=total_sample_size,
                         alpha=alpha,
                         n_jobs=n_jobs,
                         verbose=verbose,
                         message=message)
        self.bootstrap_samples = bootstrap_samples

    # for calculating power
    def estimate(self, y_true, y_pred, sample_size, stratify=None, power_statfun=None,
                 random_seed=None, n_jobs=None, verbose=None, message=None):
        super().estimate(y_true, y_pred, sample_size, stratify, power_statfun,
                         random_seed, n_jobs, verbose, message)

        # print(self.sample_size)

        resampler = Resample(stat_fun=self.power_stat_fun, sample_size=self.sample_size,
                             num_samples=self.bootstrap_samples,
                             n_jobs=self.n_jobs,
                             verbose=self.verbose, message=self.message)
        stat = resampler.bootstrap(y_true, y_pred, random_seed=random_seed, stratify=self.stratify)
        stat = np.nan_to_num(np.array(stat, dtype=np.float64), nan=1.0)
        power = np.sum(stat <= self.alpha, axis=1) / self.bootstrap_samples
        return power


def _get_sample_sizes(sample_sizes, step, current_sample_size, total_sample_size):
    if sample_sizes is not None and step is not None:
        raise AttributeError("sample_sizes and step are mutually exclusive parameters.")
    if step is None:
        step = 1
    if sample_sizes is None or isinstance(sample_sizes, Number):
        if isinstance(sample_sizes, Number):
            step = int((total_sample_size - current_sample_size) / sample_sizes)
        sample_sizes = np.arange(current_sample_size, total_sample_size + 0.5, step=step)
    # else: sample sizes are given explicitly as a list/array
    return sample_sizes


# a simple prediction method assuming that training accuracy stays constant
# this is most of the times a lower bound
def predict_power(y_true_last, y_pred_last, power_estimator,
                  total_sample_size, sample_sizes=None, step=None):
    sample_sizes = _get_sample_sizes(sample_sizes, step, len(y_pred_last), total_sample_size)

    power = power_estimator.estimate(y_true=y_true_last,
                                     y_pred=y_pred_last,
                                     sample_size=sample_sizes)

    return LearningCurveFit(data=power,
                            ns=sample_sizes,
                            scoring="power",
                            curve_type="predicted power",
                            description="predicted power",
                            method="keeplast",
                            fun=power_estimator,
                            params=None,
                            r2=np.nan)
