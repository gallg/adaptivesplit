from tqdm import tqdm
from adaptivesplit.base.utils import tqdm_joblib
import operator
from numbers import Number
import numpy as np
from joblib import Parallel, delayed
import pandas as pd
import collections
from .utils import statfun_as_callable as utils_statfun_as_callable


class _ResampleBase:
    def __init__(self, stat_fun, sample_size, num_samples, replacement, compare, first_unshuffled,
                 n_jobs, verbose, message):
        self._set_stat_fun(stat_fun)
        self.n_jobs = n_jobs
        self.num_samples = num_samples
        self.verbose = verbose
        self.message = message
        self.stats = None
        self._set_sample_size(sample_size)
        self.replacement = replacement
        self.compare = compare
        self.first_unshuffled = first_unshuffled

    def _set_stat_fun(self, stat_fun):
        self.stat_fun = utils_statfun_as_callable(stat_fun)

    def _set_sample_size(self, sample_size):
        if sample_size is None:
            self.sample_size = None
        else:
            if isinstance(sample_size, Number):
                sample_size = [sample_size]
            self.sample_size = np.array(sample_size, dtype=int)

    def _update_defaults(self, **kwargs):
        for attr, value in kwargs.items():
            if value is not None:
                if attr == 'sample_size':
                    self._set_sample_size(value)
                else:
                    setattr(self, attr, value)

    def fit_transform(self, x, y, sample_size=None, num_samples=None, replacement=None, compare=None,
                      n_jobs=None, verbose=None, random_seed=None, **kwargs):
        self._update_defaults(sample_size=sample_size, num_samples=num_samples, replacement=replacement,
                              compare=compare, n_jobs=n_jobs, verbose=verbose)
        if not isinstance(x, (list, np.ndarray)) or not isinstance(y, (list, np.ndarray)):
            raise AttributeError('x and y must be of type list or numpy.ndarray!')
        pass

    def transform(self, x, y, sample_size=None, num_samples=None, replacement=None, compare=None,
                  n_jobs=None, verbose=None, random_seed=None, **kwargs):
        self._update_defaults(sample_size=sample_size, num_samples=num_samples, replacement=replacement,
                              compare=compare, n_jobs=n_jobs, verbose=verbose)
        if self.stats is None:
            raise RuntimeError('Not yet fitted')
        pass

    def _plot(self, df, blend_hist, *args, **kwargs):
        kwargs.setdefault('kind', 'density')
        ax = df.plot(*args, **kwargs)
        if blend_hist:
            kwargs['kind'] = 'hist'
            kwargs['legend'] = False
            ax = df.plot(*args, **kwargs, density=True, alpha=blend_hist, ax=ax, color='gray')
        ax.set(xlabel=self.stat_fun.__name__)
        return ax

    def plot(self, blend_hist=0.4, *args, **kwargs):
        if self.stats is None:
            raise RuntimeError('Nothing to plot.')

        df = pd.DataFrame(self.stats, columns=[self.num_samples])
        kwargs.setdefault('legend', False)
        return self._plot(df, blend_hist, *args, **kwargs)


StatResults = collections.namedtuple("StatResults", ['statistic', 'p_value'])


class PermTest(_ResampleBase):
    """Implements a permutation test.
    
    Args:
        stat_fun (callable): 
            Statistical function used to evaluate statistical significance.
        num_samples (int): 
            Number of samples generated during permutation. Defaults to 1000.
        n_jobs (int, optional): 
            Number of jobs to run in parallel. Defaults to -1. Power calculations are parallelized 
            over the cross-validation splits. None means 1 unless in a joblib.parallel_backend context.
            -1 means using all processors.
        compare (callable): 
            User defined comparison function. Defaults to operator.ge.
        verbose (bool, optional): 
            Prints progress. Defaults to True.
        message (str, optional): 
            Message shown when permutation test starts. Defaults to "Permutation test".
    """
    def __init__(self, stat_fun, num_samples=1000, n_jobs=-1, compare=operator.ge,
                 verbose=True, message="Permutation test"):
        super().__init__(stat_fun, sample_size=None, num_samples=num_samples, compare=compare,
                         replacement=False, first_unshuffled=False, n_jobs=n_jobs,
                         verbose=verbose, message=message)
        self.p_value = np.nan

    def _update_defaults(self, **kwargs):
        if kwargs['sample_size'] is not None:
            raise AttributeError('Sample size cannot be specified for permutations tests.')
        if kwargs['replacement']:
            raise AttributeError("Can't do permutation test with replacements.")
        super()._update_defaults(**kwargs)

    def fit_transform(self, x, y, sample_size=None, num_samples=None, replacement=None, compare=None,
                      n_jobs=None, verbose=None, random_seed=None, **kwargs):
        super().fit_transform(x, y, sample_size=sample_size, num_samples=num_samples, replacement=replacement,
                              compare=compare, n_jobs=n_jobs, verbose=verbose, random_seed=random_seed, **kwargs)

        if len(y) == 0:
            self.p_value = 1
            self.stats = [np.nan]
            return self.stats

        def workhorse(_x, _y, perm_i):
            if perm_i > 0:  # 0th is the unpermuted case
                if random_seed is not None:
                    seed = random_seed + perm_i
                else:
                    seed = None
                rng = np.random.default_rng(seed)
                _y = rng.permutation(_y)

            return self.stat_fun(_x, _y, **kwargs)

        with tqdm_joblib(tqdm(desc=self.message, total=self.num_samples,
                              disable=not self.verbose)) as progress_bar:
            self.stats = Parallel(
                n_jobs=self.n_jobs)(delayed(workhorse)(x, y, p_i) for p_i in range(self.num_samples+1))

        # todo: p-values should never be 0
        self.p_value = np.sum(self.compare(self.stats[1:], self.stats[0])) / self.num_samples

        return self.stats

    def transform(self, x, y, sample_size=None, num_samples=None, replacement=None, compare=None,
                  n_jobs=None, verbose=None, random_seed=None, **kwargs):
        super().transform(x, y, sample_size=sample_size, num_samples=num_samples, replacement=replacement,
                          compare=compare, n_jobs=n_jobs, verbose=verbose, random_seed=random_seed, **kwargs)
        if len(y) == 0:
            return 1

        self.stats[0] = self.stat_fun(x, y, **kwargs)
        self.p_value = np.sum(compare(self.stats[1:], self.stats[0])) / num_samples
        return self.stats

    # human readable shortcut to fit_transform
    def test(self, x, y, num_samples=None, random_seed=None, compare=None, n_jobs=None, verbose=None):
        self.fit_transform(x, y, num_samples=num_samples, random_seed=random_seed, compare=compare,
                           n_jobs=n_jobs, verbose=verbose)                  
        return StatResults(self.stats[0], self.p_value)

    def plot(self, unpermuted_stat=None, *args, **kwargs):
        ax = super().plot(*args, **kwargs)
        if unpermuted_stat is None:
            unpermuted_stat = self.stats[0]
        ax.set(title='P =' + str(self.p_value))
        ax.axvline(unpermuted_stat)
        return ax


class Resample(_ResampleBase):
    """Implements re-sampling.
    
    Args:
        stat_fun (callable): 
            User defined statistical function.
        sample_size (int): 
            Current sample size.
        stratify (int): 
            For classification tasks. If not None, use stratified sampling to account for class labels imbalance. 
            Defaults to None.
        num_samples (int, optional): 
            Number of samples. Defaults to 1000.
        replacement (bool, optional): 
            Whether or not to sample with replacement. Defaults to True.
        first_unshuffled (bool, optional): 
            Whether or not to shuffle the samples first. Defaults to False.
        n_jobs (int, optional): 
            Number of jobs to run in parallel. Defaults to -1. Power calculations are parallelized 
            over the cross-validation splits. None means 1 unless in a joblib.parallel_backend context.
            -1 means using all processors.
        verbose (bool, optional): 
            Print progress. Defaults to True.
        message (str, optional): 
            Message shown when resampling starts. Defaults to "Resampling".
    """
    def __init__(
            self,
            stat_fun,
            sample_size,
            stratify=None,
            num_samples=1000,
            replacement=True,
            first_unshuffled=False,
            n_jobs=-1,
            verbose=True, message="Resampling"
    ):
        super().__init__(
            stat_fun,
            sample_size=sample_size,
            num_samples=num_samples,
            replacement=replacement,
            first_unshuffled=first_unshuffled,
            n_jobs=n_jobs,
            verbose=verbose,
            message=message,
            compare=None
        )
        self.stratify = stratify

    def _update_defaults(self, **kwargs):
        if 'compare' in kwargs and kwargs['compare'] is not None:
            raise AttributeError("The parameter 'compare' is not usable for resampling")
        super()._update_defaults(**kwargs)

    def fit_transform(self, x, y, stratify=None, sample_size=None, num_samples=None, replacement=None, compare=None,
                      n_jobs=None, verbose=None, random_seed=None, **kwargs):
        super().fit_transform(x, y, sample_size=sample_size, num_samples=num_samples, replacement=replacement,
                              compare=compare, n_jobs=n_jobs, verbose=verbose, random_seed=random_seed, **kwargs)
        self._update_defaults(stratify=stratify)

        if self.stratify is None:
            self.stratify = np.zeros_like(y)  # a single strata
        if isinstance(stratify, int):
            self.stratify = np.digitize(
                y, bins=np.quantile(y, np.linspace(0, 1, num=stratify + 1, endpoint=True))
            )

        # workhorse
        def workhorse(_i, _n, _s):

            if random_seed is not None:
                state = int(random_seed + _n * _s)
            else:
                state = random_seed

            rng = np.random.default_rng(state)  # according to latest numpy recommendations
            # initialize with random indices, this will be overwritten, except if n_strata does not sum up to _n
            # in this case the results will be padded with random indices #todo: not good if replacement = False
            resampled_indices = np.array([], dtype=int)

            strata, counts = np.unique(self.stratify, return_counts=True)
            sample_ratio = _n / len(y)
            n_strata = np.trunc(sample_ratio * counts).astype(int)
            for i in range(int(_n - np.sum(n_strata))):
                n_strata[np.random.randint(len(n_strata), size=1)] += 1
            for i_stratum, stratum in enumerate(strata):
                idx_strata = np.argwhere(self.stratify == stratum).flatten()
                if self.first_unshuffled and _s == 0 and not self.replacement:
                    resampled_indices = np.hstack((resampled_indices, idx_strata[: n_strata[i_stratum]]))  # no shuffle
                else:
                    resampled_indices = np.hstack((resampled_indices,
                                                   rng.choice(idx_strata,
                                                              size=n_strata[i_stratum],
                                                              replace=self.replacement)))

            _x = x[resampled_indices]
            _y = y[resampled_indices]

            stat = self.stat_fun(_x, _y, random_seed=random_seed, **kwargs)
            return _i, _n, _s, stat

        with tqdm_joblib(tqdm(desc=self.message,
                              total=self.num_samples * len(self.sample_size),
                              disable=not self.verbose)) as progress_bar:

            i, n, s, stats = zip(
                *Parallel(n_jobs=self.n_jobs)(
                    delayed(workhorse)(i, n, s) for i, n in enumerate(self.sample_size) for s in
                    range(self.num_samples))
            )
        self.stats = []
        for samp_i, samp in enumerate(self.sample_size):
            self.stats.append(np.array(stats, dtype=object)[np.array(i) == samp_i])

        self.stats = np.array(self.stats)

        return self.stats

    def transform(self, x, y, stratify=None, sample_size=None, num_samples=None, replacement=None, compare=None,
                  n_jobs=None, verbose=None, random_seed=None, **kwargs):
        super().transform(
            x, y, stratify=stratify, sample_size=sample_size, num_samples=num_samples, replacement=replacement,
            compare=compare, n_jobs=n_jobs, verbose=verbose, random_seed=random_seed, **kwargs
        )
        return self.stats

    def bootstrap(self, x, y, stratify=None, sample_size=None, num_samples=None,
                  random_seed=None, n_jobs=None, verbose=None,
                  *args, **kwargs):
        return self.fit_transform(x, y, stratify=stratify, sample_size=sample_size,
                                  num_samples=num_samples, replacement=True,
                                  random_seed=random_seed, n_jobs=n_jobs, verbose=verbose,
                                  *args, **kwargs)

    def subsample(self, x, y, stratify=None, sample_size=None, num_samples=None,
                  random_seed=None, n_jobs=None, verbose=None,
                  *args, **kwargs):
        return self.fit_transform(x, y, stratify=stratify, sample_size=sample_size,
                                  num_samples=num_samples, replacement=False,
                                  random_seed=random_seed, n_jobs=n_jobs,
                                  verbose=verbose, *args, **kwargs)

    def plot(self, blend_hist=0.2, *args, **kwargs):
        if isinstance(self.sample_size, Number):
            sample_size = [self.sample_size]
        else:
            sample_size = self.sample_size

        df = pd.DataFrame(np.array(self.stats).transpose(), columns=sample_size)
        return self._plot(df, blend_hist, *args, **kwargs)
