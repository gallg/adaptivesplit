import warnings

import numpy as np
import pandas as pd
from pandas.io.formats.format import format_percentiles
from scipy.optimize import curve_fit


class LearningCurve:
    """Class for storing and handling learning curves.
    Wraps a pandas dataframe with rows for learning curves and columns for sample sizes
    It can handle multiple learning curves (e.g. for different bootstrap samples).
    
    Args:
        df (array-like, pandas.DataFrame): 
            Initialize by a DataFrame, already being in the correct format. Defaults to None.
        ns (int): 
            Alias for number of samples (sample sizes). Defaults to None.
        data (np.ndarray): 
            Array of shape (ns, n_curves). Contains score value per sample size. Defaults to None.
        scoring (str, callable, list, tuple or dict): 
            Scikit-learn-like score to evaluate the performance of the cross-validated model on the test set.
            If scoring represents a single score, one can use:

            - a single string (see The scoring parameter: defining model evaluation rules);
            - a callable (see Defining your scoring strategy from metric functions) that returns a single value.

            If scoring represents multiple scores, one can use:

            - a list or tuple of unique strings;
            - a callable returning a dictionary where the keys are the metric names and the values are
              the metric scores;
            - a dictionary with metric names as keys and callables a values.

            If None, the estimator’s score method is used. Defaults to None.
        curve_names (str, optional): 
            Curve names metadata. Defaults to None.
        curve_type (str, optional): 
            Curve type metadata. Defaults to None.
        description (str, optional): 
            Curve description. Defaults to None. 
    """

    def __init__(self, df=None, ns=None, data=None, scoring=None,
                 curve_names=None, curve_type=None, description=None):

        if data is not None:
            if ns is None:
                raise RuntimeError("Sample sizes must be explicitly provided.")
            if scoring is None:
                raise RuntimeError("Score must be explicitly provided.")
            if isinstance(data, list):
                data = np.array(data)
            if len(data.shape) == 1:
                data = data.reshape(-1, len(ns))
            if data.shape[1] != len(ns):
                raise RuntimeError('learning_curve shape does not match ns', str(data.shape), str(len(ns)))

            self.df = pd.DataFrame(data=data, index=None, columns=ns).transpose()
            if curve_names is not None:
                self.df.columns = curve_names
        else:
            self.df = df
            if ns is not None:
                self.df.index = np.array(ns, dtype=int)

        self.df.index.name = "sample size"
        self.df.columns.name = curve_type

        self.scorer = scoring
        if isinstance(scoring, str):
            self.score_name = scoring
        elif callable(scoring):
            self.score_name = scoring.__name__
        else:
            raise RuntimeError("Unknown score.")

        self.scorer = scoring
        self.description = description

    def dump(self):
        """Dump learning curve data."""
        print(self.df)
        print("score:", self.scorer)
        print("description:", self.description)

    def __str__(self):
        return self.df.columns.name

    def plot(self, add=None, *args, **kwargs):
        ax = self.df.plot(*args, **kwargs)
        if isinstance(self, LearningCurveStat):
            ax.fill_between(self.df.index, self.df.lower, y2=self.df.upper, alpha=0.2, color=ax.get_lines()[-1].get_c())

        legend = [self.df.columns.name]

        if add is not None:
            if not isinstance(add, (list, tuple, np.ndarray)):
                add = [add]
            for lc in add:
                if isinstance(lc, LearningCurve):
                    ax = lc.plot(ax=ax, *args, **kwargs)
                    legend += [lc.df.columns.name]
                else:
                    raise RuntimeError("Only LearningCurve objects can be added to plot")

        if "legend" not in kwargs or kwargs["legend"]:
            ax.legend(legend)

        ax.set_ylabel(self.score_name)
        ax.set_title(self.df.columns.name)

        return ax

    def stat(self, mid="mean", ci="95%"):
        """Return simple descriptive learning curve stats (e.g for plotting purposes).
        Stat names should be given according to pandas.describe().
        Most common choices: 'mean', '50%', 'std', '95%'.
        
        
        Args:
            mid (str):
                Stat to calculate mid value.
            ci (str):
                Confidence level.

        Extra keywords:
            'stderr' gives mid +- stderr/2
        """
        lower = upper = 'n/a'

        percentiles = []
        if mid[-1] == '%':
            percentiles.append(float(mid[:-1]) / 100)
        if ci[-1] == '%':
            cinum = float(ci[:-1]) / 100
            tail = (1 - cinum) / 2
            lowernum = 0 + tail
            lower = format_percentiles([lowernum])[0]
            uppernum = 1 - tail
            upper = format_percentiles([uppernum])[0]
            percentiles.append(lowernum)
            percentiles.append(uppernum)

        self.df = self.df.apply(pd.to_numeric)
        stat = self.df.transpose().describe(percentiles=percentiles)

        if ci == 'stderr':
            # stderr
            stderr = self.df.transpose().sem()
            lower = 'stderr_lower'
            stat = stat.append(pd.Series(stat.loc[mid] - stderr / 2.0, name='stderr_lower'))
            upper = 'stderr_upper'
            stat = stat.append(pd.Series(stat.loc[mid] + stderr / 2.0, name='stderr_upper'))

        if ci == 'std':
            lower = 'std_lower'
            stat = stat.append(pd.Series(stat.loc[mid] - stat.loc['std'] / 2, name='std_lower'))
            upper = 'std_upper'
            stat = stat.append(pd.Series(stat.loc[mid] + stat.loc['std'] / 2, name='std_upper'))
        # rename
        stat = stat.rename(dict(zip([mid, lower, upper], ['mid', 'lower', 'upper'])))
        if self.df.columns.name is None:
            curve_type = mid + ' & ' + ci
        else:
            curve_type = self.df.columns.name + ' ' + mid + ' and ' + ci

        return LearningCurveStat(stat.loc[['mid', 'upper', 'lower']].transpose(),
                                 scoring=self.scorer,
                                 curve_type=curve_type,
                                 description=dict(zip(['mid', 'upper', 'lower'], [mid, upper, lower])),
                                 ci=ci)

    def fit(self, extend=[], new_x=None, method='nlls', fun=lambda x, a, b: a - b / np.log(x), init=[1, 1.6],
            **kwargs):

        extend = np.array(extend, dtype=int)

        if new_x is None:
            fitted = np.zeros(len(self.df) + len(extend))
            new_x = np.array(self.df.index.tolist() + extend.tolist())
        else:
            if len(extend) != 0:
                raise AttributeError("'extend' and new_x are mutually exclusive.")
            if method == 'keeplast':
                raise AttributeError("'new_x' can't be used with method == 'keeplast'. Consider using 'extend'.")
            fitted = np.zeros(len(new_x))

        if method == 'keeplast':
            fitted[:len(self.df)] = self.df.transpose().mean().values
            fitted[len(self.df):] = fitted[len(self.df) - 1]
            fun = None
            r2 = 1
            name = method + f'(r2={r2:.2f})'
            popt = None

        elif method == 'non-linear_least_squares' or method == 'nlls':
            x = np.repeat(self.df.index.values, len(self.df.columns))
            y = self.df.values.flatten()
            popt, pcov = curve_fit(f=fun,
                                   full_output=False,
                                   xdata=x[~ np.isnan(y)],
                                   ydata=y[~ np.isnan(y)],
                                   p0=init)
            fitted = fun(new_x, *popt)
            full_fitted = fun(x[~ np.isnan(y)], *popt)
            r2 = np.corrcoef(y[~ np.isnan(y)], full_fitted)[0, 1] ** 2
            name = method + f'(r2={r2:.2f})'
        else:
            raise AttributeError('No such learning curve fitting method.')

        if self.df.columns.name is None:
            self.df.columns.name = ''
        return LearningCurveFit(data=fitted,
                                ns=new_x,  # todo fixme
                                scoring=self.scorer,
                                curve_type=self.df.columns.name + ' fit ' + name,
                                description='fit: ' + name,
                                method=method,
                                fun=fun,
                                params=popt,
                                r2=r2)


class LearningCurveStat(LearningCurve):

    def __init__(self, *args, ci="", **kwargs):
        super().__init__(*args, **kwargs)
        self.ci = ci

    def plot(self, add=None, *args, **kwargs):
        kwargs['y'] = 'mid'
        ax = super().plot(add=add, *args, **kwargs)
        if add is None:
            ax.legend([list(self.description.values())[0], self.ci])
        return ax

    def fit(self, extend=[], new_x=None, method='keeplast', fun=lambda x, a, b: a - b / np.log(x), init=[1, 1.6],
            **kwargs):
        warnings.warn("Fitting statistical curves is NOT recommended!")
        super().fit(self, extend=extend, new_x=new_x, method=method,
                    fun=fun, init=init, **kwargs)


class LearningCurveFit(LearningCurve):

    def __init__(self, *args, method, fun, params, r2, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method
        self.fun = fun
        self.params = params
        self.r2 = r2

    def extrapolate(self, new_x):
        if self.method == 'keeplast':
            raise RuntimeError("Can't extrapolate curve fit with method 'keeplast'.")
        elif self.method == 'nlls':
            self.df = pd.DataFrame(data=self.fun(new_x, *self.params), index=np.array(new_x, dtype=int))
        else:
            raise AttributeError('No such learning curve fitting method.')
        return self

    def fit(self, extend=[], new_x=None, method='keeplast', fun=lambda x, a, b: a - b / np.log(x), init=[1, 1.6],
            **kwargs):
        raise RuntimeError("Curve is already fitted!")

    def plot(self, add=None, *args, **kwargs):
        kwargs.setdefault('linestyle', ':')
        kwargs.setdefault('linewidth', 2)
        ax = super().plot(add=add, *args, **kwargs)
        return ax


# shorthand for plotting
def plot_learning_curves(*args, **kwargs):
    return args[0].plot(add=[lc for lc in args[1:]], **kwargs)


def plot_stat_learning_curves(*args, mid='mean', ci='95%', **kwargs):
    return args[0].stat(mid=mid, ci=ci).plot(add=[lc.stat(mid=mid, ci=ci) for lc in args[1:]], **kwargs)
